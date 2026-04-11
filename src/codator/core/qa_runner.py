"""QA Runner - comprehensive automated testing for Ethos OS NAS.

Systematically tests all Ethos applications via API, browser UI, workflow
validation, event log monitoring, and network diagnostics, then creates
tickets for any issues found.
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import re as _re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from codator.infrastructure.tools.ethos_ticket_tool import EthosClient

logger = logging.getLogger(__name__)


# -- Workflow definitions for deep functional testing per app ----------------
# Each workflow has steps: wait, screenshot, check_errors, evaluate, api_call,
# assert_elements, assert_text, sleep, api_call_dynamic

ETHOS_APP_WORKFLOWS: dict[str, dict[str, Any]] = {
    "dashboard": {
        "name": "Dashboard",
        "steps": [
            {"action": "wait", "selector": ".widget, .dashboard, [class*=dashboard]", "timeout": 8000},
            {"action": "screenshot", "label": "dashboard_loaded"},
            {"action": "assert_elements", "selector": ".widget, .card, [class*=widget]",
             "min_count": 1, "description": "Dashboard should show at least one widget"},
            {"action": "check_errors"},
            {"action": "evaluate", "js": "document.querySelector('.refresh-btn, [title*=Refresh], [class*=refresh]')?.click()"},
            {"action": "sleep", "ms": 2000},
            {"action": "screenshot", "label": "dashboard_after_refresh"},
            {"action": "check_errors"},
        ],
    },
    "file-manager": {
        "name": "File Manager",
        "steps": [
            {"action": "wait", "selector": ".file-list, .file-browser, [class*=file]", "timeout": 8000},
            {"action": "screenshot", "label": "filemanager_initial"},
            {"action": "check_errors"},
            {"action": "evaluate", "js": "(function(){var n=document.querySelector('[data-path=\"/home\"],.breadcrumb a');if(n){n.click();return 'clicked';}return 'not found';})()"},
            {"action": "sleep", "ms": 2000},
            {"action": "screenshot", "label": "filemanager_home"},
            {"action": "api_call", "method": "POST", "path": "/files/mkdir",
             "body": {"path": "/home/marcin/_qa_test_folder"}, "label": "create_test_folder"},
            {"action": "sleep", "ms": 1000},
            {"action": "evaluate", "js": "document.querySelector('[class*=refresh],.refresh-btn')?.click()"},
            {"action": "sleep", "ms": 2000},
            {"action": "screenshot", "label": "filemanager_with_test_folder"},
            {"action": "assert_text", "text": "_qa_test_folder",
             "description": "Created test folder should appear in file list"},
            {"action": "api_call", "method": "POST", "path": "/files/write",
             "body": {"path": "/home/marcin/_qa_test_folder/test.txt",
                      "content": "QA test file created by codator"},
             "label": "create_test_file"},
            {"action": "api_call", "method": "GET",
             "path": "/files/read?path=/home/marcin/_qa_test_folder/test.txt",
             "expect_contains": "QA test file", "label": "verify_test_file"},
            {"action": "api_call", "method": "POST", "path": "/files/rename",
             "body": {"path": "/home/marcin/_qa_test_folder/test.txt",
                      "new_name": "test_renamed.txt"},
             "label": "rename_test_file"},
            {"action": "check_errors"},
            {"action": "api_call", "method": "POST", "path": "/files/delete",
             "body": {"path": "/home/marcin/_qa_test_folder"}, "label": "cleanup_folder",
             "is_cleanup": True},
        ],
    },
    "docker-manager": {
        "name": "Docker Manager",
        "steps": [
            {"action": "wait", "selector": ".docker, .container-list, [class*=docker]", "timeout": 8000},
            {"action": "screenshot", "label": "docker_initial"},
            {"action": "check_errors"},
            {"action": "api_call", "method": "GET", "path": "/docker/containers", "label": "list_containers"},
            {"action": "api_call", "method": "GET", "path": "/docker/images", "label": "list_images"},
            {"action": "api_call", "method": "GET", "path": "/docker/system", "label": "docker_system_info"},
            {"action": "api_call", "method": "GET", "path": "/docker/projects", "label": "docker_projects"},
            {"action": "evaluate", "js": "(function(){var tabs=document.querySelectorAll('.tab,[role=tab],[class*=tab]');tabs.forEach(function(t,i){setTimeout(function(){t.click()},i*500)});return 'Clicked '+tabs.length+' tabs';})()"},
            {"action": "sleep", "ms": 3000},
            {"action": "screenshot", "label": "docker_after_tabs"},
            {"action": "check_errors"},
        ],
    },
    "storage-manager": {
        "name": "Storage Manager",
        "steps": [
            {"action": "wait", "selector": ".storage, .drives, [class*=storage]", "timeout": 8000},
            {"action": "screenshot", "label": "storage_initial"},
            {"action": "check_errors"},
            {"action": "api_call", "method": "GET", "path": "/storage/drives", "label": "drives"},
            {"action": "api_call", "method": "GET", "path": "/storage/health", "label": "health"},
            {"action": "api_call", "method": "GET", "path": "/storage/pool/list", "label": "pools"},
            {"action": "api_call", "method": "GET", "path": "/storage/samba/shares", "label": "samba"},
            {"action": "api_call", "method": "GET", "path": "/storage/nfs/exports", "label": "nfs"},
            {"action": "api_call", "method": "GET", "path": "/storage/smart/schedule", "label": "smart"},
            {"action": "api_call", "method": "GET", "path": "/storage/app-usage", "label": "usage"},
            {"action": "evaluate", "js": "(function(){var tabs=document.querySelectorAll('.tab,[role=tab],nav a,.nav-item');tabs.forEach(function(t,i){setTimeout(function(){t.click()},i*800)});return 'Navigated '+tabs.length+' tabs';})()"},
            {"action": "sleep", "ms": 4000},
            {"action": "screenshot", "label": "storage_after_navigation"},
            {"action": "check_errors"},
        ],
    },
    "users": {
        "name": "User Management",
        "steps": [
            {"action": "wait", "selector": ".user-list, .users, [class*=user]", "timeout": 8000},
            {"action": "screenshot", "label": "users_initial"},
            {"action": "check_errors"},
            {"action": "api_call", "method": "GET", "path": "/users/list", "label": "user_list"},
            {"action": "api_call", "method": "GET", "path": "/users/groups", "label": "groups"},
            {"action": "api_call", "method": "GET", "path": "/users/password-policy", "label": "pw_policy"},
            {"action": "evaluate", "js": "(function(){var r=document.querySelectorAll('.user-item,.user-row,tr[data-user],tbody tr');if(r.length>0){r[0].click();return 'Clicked first user';}return 'No users';})()"},
            {"action": "sleep", "ms": 1500},
            {"action": "screenshot", "label": "users_detail"},
            {"action": "check_errors"},
        ],
    },
    "network": {
        "name": "Network",
        "steps": [
            {"action": "wait", "selector": ".network, .interfaces, [class*=network]", "timeout": 8000},
            {"action": "screenshot", "label": "network_initial"},
            {"action": "check_errors"},
            {"action": "api_call", "method": "GET", "path": "/network/interfaces", "label": "interfaces"},
            {"action": "api_call", "method": "GET", "path": "/network/bonds", "label": "bonds"},
            {"action": "api_call", "method": "GET", "path": "/network/wifi/status", "label": "wifi"},
            {"action": "evaluate", "js": "(function(){var c=document.querySelectorAll('.interface-card,.interface-item,.card');c.forEach(function(e,i){setTimeout(function(){e.click()},i*600)});return 'Clicked '+c.length+' cards';})()"},
            {"action": "sleep", "ms": 2000},
            {"action": "screenshot", "label": "network_after_clicks"},
            {"action": "check_errors"},
        ],
    },
    "tickets": {
        "name": "Tickets",
        "steps": [
            {"action": "wait", "selector": ".tickets, .kanban, .board, [class*=ticket]", "timeout": 8000},
            {"action": "screenshot", "label": "tickets_initial"},
            {"action": "check_errors"},
            {"action": "api_call", "method": "GET", "path": "/tickets/projects", "label": "projects"},
            {"action": "api_call", "method": "POST", "path": "/tickets/tickets",
             "body": {"title": "[QA-TEST] Automated test ticket", "description": "Created by codator QA. Delete me.",
                      "type": "task", "priority": "low", "column": "Backlog"},
             "label": "create_test_ticket", "save_result": "test_ticket"},
            {"action": "sleep", "ms": 1500},
            {"action": "screenshot", "label": "tickets_after_create"},
            {"action": "api_call_dynamic", "method": "PUT",
             "path_template": "/tickets/tickets/{test_ticket.id}",
             "body": {"title": "[QA-TEST] Updated test ticket", "column": "Do zrobienia"},
             "label": "update_test_ticket"},
            {"action": "sleep", "ms": 1000},
            {"action": "screenshot", "label": "tickets_after_update"},
            {"action": "api_call_dynamic", "method": "DELETE",
             "path_template": "/tickets/tickets/{test_ticket.id}",
             "label": "delete_test_ticket", "is_cleanup": True},
            {"action": "check_errors"},
        ],
    },
    "terminal": {
        "name": "Terminal",
        "steps": [
            {"action": "wait", "selector": ".terminal, .xterm, [class*=terminal]", "timeout": 8000},
            {"action": "screenshot", "label": "terminal_loaded"},
            {"action": "check_errors"},
            {"action": "evaluate", "js": "(function(){var t=document.querySelector('.xterm-helper-textarea,textarea,.terminal-input');if(t){t.focus();return 'focused';}return 'not found';})()"},
            {"action": "sleep", "ms": 1000},
            {"action": "screenshot", "label": "terminal_ready"},
            {"action": "check_errors"},
        ],
    },
    "backup": {
        "name": "Backup",
        "steps": [
            {"action": "wait", "selector": ".backup, [class*=backup]", "timeout": 8000},
            {"action": "screenshot", "label": "backup_initial"},
            {"action": "check_errors"},
            {"action": "api_call", "method": "GET", "path": "/backup/status", "label": "status"},
            {"action": "api_call", "method": "GET", "path": "/backup/profiles", "label": "profiles"},
            {"action": "api_call", "method": "GET", "path": "/backup/history", "label": "history"},
            {"action": "api_call", "method": "GET", "path": "/backup/snapshots", "label": "snapshots"},
            {"action": "api_call", "method": "GET", "path": "/backup/scheduled-backups", "label": "scheduled"},
            {"action": "evaluate", "js": "(function(){var t=document.querySelectorAll('.tab,[role=tab],nav a');t.forEach(function(e,i){setTimeout(function(){e.click()},i*600)});return t.length+' tabs';})()"},
            {"action": "sleep", "ms": 3000},
            {"action": "screenshot", "label": "backup_after_tabs"},
            {"action": "check_errors"},
        ],
    },
    "services": {
        "name": "Services",
        "steps": [
            {"action": "wait", "selector": ".services, [class*=service]", "timeout": 8000},
            {"action": "screenshot", "label": "services_initial"},
            {"action": "check_errors"},
            {"action": "api_call", "method": "GET", "path": "/services/list", "label": "services"},
            {"action": "evaluate", "js": "(function(){var r=document.querySelectorAll('.service-item,.service-row,tr');if(r.length){r[0].click();return 'clicked';}return 'none';})()"},
            {"action": "sleep", "ms": 1500},
            {"action": "screenshot", "label": "services_detail"},
            {"action": "check_errors"},
        ],
    },
    "security-advisor": {
        "name": "Security Advisor",
        "steps": [
            {"action": "wait", "selector": ".security, [class*=security]", "timeout": 8000},
            {"action": "screenshot", "label": "security_initial"},
            {"action": "check_errors"},
            {"action": "api_call", "method": "GET", "path": "/security/scan/status", "label": "scan_status"},
            {"action": "api_call", "method": "GET", "path": "/security/scan/results", "label": "scan_results"},
            {"action": "check_errors"},
        ],
    },
    "firewall": {
        "name": "Firewall",
        "steps": [
            {"action": "wait", "selector": ".firewall, [class*=firewall]", "timeout": 8000},
            {"action": "screenshot", "label": "firewall_initial"},
            {"action": "check_errors"},
            {"action": "api_call", "method": "GET", "path": "/firewall/status", "label": "status"},
            {"action": "api_call", "method": "GET", "path": "/firewall/rules", "label": "rules"},
            {"action": "api_call", "method": "GET", "path": "/firewall/banned", "label": "banned"},
            {"action": "check_errors"},
        ],
    },
    "wireguard": {
        "name": "WireGuard VPN",
        "steps": [
            {"action": "wait", "selector": ".wireguard, [class*=wireguard], [class*=vpn]", "timeout": 8000},
            {"action": "screenshot", "label": "wireguard_initial"},
            {"action": "check_errors"},
            {"action": "api_call", "method": "GET", "path": "/wireguard/status", "label": "status"},
            {"action": "api_call", "method": "GET", "path": "/wireguard/peers", "label": "peers"},
            {"action": "check_errors"},
        ],
    },
    "gallery": {
        "name": "Gallery",
        "steps": [
            {"action": "wait", "selector": ".gallery, [class*=gallery]", "timeout": 8000},
            {"action": "screenshot", "label": "gallery_initial"},
            {"action": "check_errors"},
            {"action": "api_call", "method": "GET", "path": "/gallery/albums", "label": "albums"},
            {"action": "evaluate", "js": "(function(){var i=document.querySelector('.album,.photo-item,.thumbnail,.gallery-item');if(i){i.click();return 'clicked';}return 'none';})()"},
            {"action": "sleep", "ms": 2000},
            {"action": "screenshot", "label": "gallery_detail"},
            {"action": "check_errors"},
        ],
    },
    "video-station": {
        "name": "Video Station",
        "steps": [
            {"action": "wait", "selector": ".video, [class*=video]", "timeout": 8000},
            {"action": "screenshot", "label": "video_initial"},
            {"action": "check_errors"},
            {"action": "api_call", "method": "GET", "path": "/video/library", "label": "library"},
            {"action": "check_errors"},
        ],
    },
    "resource-monitor": {
        "name": "Resource Monitor",
        "steps": [
            {"action": "wait", "selector": ".resources, [class*=resource], [class*=monitor]", "timeout": 8000},
            {"action": "screenshot", "label": "resources_initial"},
            {"action": "check_errors"},
            {"action": "api_call", "method": "GET", "path": "/resources/all", "label": "all"},
            {"action": "api_call", "method": "GET", "path": "/resources/processes", "label": "processes"},
            {"action": "assert_elements", "selector": ".chart, canvas, svg, [class*=chart]",
             "min_count": 1, "description": "Should render resource charts"},
            {"action": "check_errors"},
        ],
    },
    "packages": {
        "name": "Package Manager",
        "steps": [
            {"action": "wait", "selector": ".packages, [class*=package]", "timeout": 8000},
            {"action": "screenshot", "label": "packages_initial"},
            {"action": "check_errors"},
            {"action": "api_call", "method": "GET", "path": "/packages/installed", "label": "installed"},
            {"action": "api_call", "method": "GET", "path": "/packages/upgradable", "label": "upgradable"},
            {"action": "check_errors"},
        ],
    },
    "updates": {
        "name": "System Updates",
        "steps": [
            {"action": "wait", "selector": ".updates, [class*=update]", "timeout": 8000},
            {"action": "screenshot", "label": "updates_initial"},
            {"action": "check_errors"},
            {"action": "api_call", "method": "GET", "path": "/updates/status", "label": "status"},
            {"action": "check_errors"},
        ],
    },
    "fail2ban": {
        "name": "Intrusion Protection",
        "steps": [
            {"action": "wait", "selector": ".fail2ban, [class*=fail2ban]", "timeout": 8000},
            {"action": "screenshot", "label": "fail2ban_initial"},
            {"action": "check_errors"},
            {"action": "api_call", "method": "GET", "path": "/fail2ban/status", "label": "status"},
            {"action": "api_call", "method": "GET", "path": "/fail2ban/jails", "label": "jails"},
            {"action": "check_errors"},
        ],
    },
    "antivirus": {
        "name": "Antivirus",
        "steps": [
            {"action": "wait", "selector": ".antivirus, [class*=antivirus]", "timeout": 8000},
            {"action": "screenshot", "label": "antivirus_initial"},
            {"action": "check_errors"},
            {"action": "api_call", "method": "GET", "path": "/antivirus/status", "label": "status"},
            {"action": "check_errors"},
        ],
    },
    "ai-chat": {
        "name": "AI Chat",
        "steps": [
            {"action": "wait", "selector": ".chat, [class*=chat]", "timeout": 8000},
            {"action": "screenshot", "label": "aichat_initial"},
            {"action": "check_errors"},
            {"action": "assert_elements", "selector": "textarea, .chat-input, [class*=input]",
             "min_count": 1, "description": "Chat should have input field"},
            {"action": "check_errors"},
        ],
    },
    "sticky-notes": {
        "name": "Sticky Notes",
        "steps": [
            {"action": "wait", "selector": ".stickynotes, .notes, [class*=sticky]", "timeout": 8000},
            {"action": "screenshot", "label": "stickynotes_initial"},
            {"action": "check_errors"},
            {"action": "api_call", "method": "POST", "path": "/stickynotes/notes",
             "body": {"title": "QA Test Note", "content": "Created by codator QA", "color": "#ffff88"},
             "label": "create_note", "save_result": "test_note"},
            {"action": "sleep", "ms": 1000},
            {"action": "screenshot", "label": "stickynotes_after_create"},
            {"action": "api_call_dynamic", "method": "DELETE",
             "path_template": "/stickynotes/notes/{test_note.id}",
             "label": "delete_note", "is_cleanup": True},
            {"action": "check_errors"},
        ],
    },
    "cron": {
        "name": "Scheduler",
        "steps": [
            {"action": "wait", "selector": ".cron, .scheduler, [class*=cron]", "timeout": 8000},
            {"action": "screenshot", "label": "cron_initial"},
            {"action": "check_errors"},
            {"action": "api_call", "method": "GET", "path": "/cron/jobs", "label": "jobs"},
            {"action": "check_errors"},
        ],
    },
    "domains-manager": {
        "name": "Domains & SSL",
        "steps": [
            {"action": "wait", "selector": ".domains, [class*=domain]", "timeout": 8000},
            {"action": "screenshot", "label": "domains_initial"},
            {"action": "check_errors"},
            {"action": "api_call", "method": "GET", "path": "/domains/list", "label": "domains"},
            {"action": "api_call", "method": "GET", "path": "/domains/ssl", "label": "ssl"},
            {"action": "check_errors"},
        ],
    },
    "sharing-samba": {
        "name": "Samba Sharing",
        "steps": [
            {"action": "wait", "selector": ".samba, .sharing, [class*=samba]", "timeout": 8000},
            {"action": "screenshot", "label": "samba_initial"},
            {"action": "check_errors"},
            {"action": "api_call", "method": "GET", "path": "/storage/samba/shares", "label": "shares"},
            {"action": "api_call", "method": "GET", "path": "/storage/samba/status", "label": "status"},
            {"action": "check_errors"},
        ],
    },
    "surveillance": {
        "name": "Surveillance",
        "steps": [
            {"action": "wait", "selector": ".surveillance, .camera, [class*=surveillance]", "timeout": 8000},
            {"action": "screenshot", "label": "surveillance_initial"},
            {"action": "check_errors"},
            {"action": "api_call", "method": "GET", "path": "/surveillance/cameras", "label": "cameras"},
            {"action": "check_errors"},
        ],
    },
    "power": {
        "name": "Power Management",
        "steps": [
            {"action": "wait", "selector": ".power, [class*=power]", "timeout": 8000},
            {"action": "screenshot", "label": "power_initial"},
            {"action": "check_errors"},
            {"action": "api_call", "method": "GET", "path": "/power/schedule", "label": "schedule"},
            {"action": "check_errors"},
        ],
    },
    "ups": {
        "name": "UPS",
        "steps": [
            {"action": "wait", "selector": ".ups, [class*=ups]", "timeout": 8000},
            {"action": "screenshot", "label": "ups_initial"},
            {"action": "check_errors"},
            {"action": "api_call", "method": "GET", "path": "/ups/status", "label": "status"},
            {"action": "check_errors"},
        ],
    },
    "rollback": {
        "name": "System Rollback",
        "steps": [
            {"action": "wait", "selector": ".rollback, [class*=rollback]", "timeout": 8000},
            {"action": "screenshot", "label": "rollback_initial"},
            {"action": "check_errors"},
            {"action": "api_call", "method": "GET", "path": "/rollback/snapshots", "label": "snapshots"},
            {"action": "check_errors"},
        ],
    },
    "download-manager": {
        "name": "Download Manager",
        "steps": [
            {"action": "wait", "selector": ".downloads, [class*=download]", "timeout": 8000},
            {"action": "screenshot", "label": "downloads_initial"},
            {"action": "check_errors"},
            {"action": "api_call", "method": "GET", "path": "/downloads/list", "label": "list"},
            {"action": "check_errors"},
        ],
    },
}


# -- All known Ethos API endpoints by app (for smoke testing) ----------------

ETHOS_APP_ENDPOINTS: dict[str, list[str]] = {
    "auth": ["/auth/verify"],
    "dashboard": ["/dashboard/summary"],
    "system": ["/system/info"],
    "apps": ["/apps"],
    "users": ["/users/list", "/users/groups", "/users/password-policy"],
    "storage": [
        "/storage/drives", "/storage/health", "/storage/pool/list",
        "/storage/samba/shares", "/storage/samba/status",
        "/storage/nfs/exports", "/storage/nfs/status",
        "/storage/ftp/status", "/storage/sftp/status",
        "/storage/webdav/shares", "/storage/webdav/status",
        "/storage/smart/schedule", "/storage/app-usage",
        "/storage/maintenance/status", "/storage/maintenance/history?limit=20",
    ],
    "files": ["/files/list?path=/home"],
    "network": [
        "/network/interfaces", "/network/bonds",
        "/network/wifi/status", "/network/wifi/saved",
        "/network/ap/status",
    ],
    "docker": [
        "/docker/status", "/docker/containers",
        "/docker/images", "/docker/projects", "/docker/system",
    ],
    "backup": [
        "/backup/status", "/backup/profiles", "/backup/history",
        "/backup/snapshots", "/backup/scheduled-backups",
        "/backup/btrfs-snapshots",
    ],
    "services": ["/services/list"],
    "packages": ["/packages/installed", "/packages/stats", "/packages/upgradable"],
    "firewall": ["/firewall/status", "/firewall/rules", "/firewall/banned"],
    "fail2ban": ["/fail2ban/status", "/fail2ban/jails"],
    "wireguard": ["/wireguard/status", "/wireguard/peers"],
    "antivirus": ["/antivirus/status"],
    "cron": ["/cron/jobs"],
    "updates": ["/updates/status", "/updates/history"],
    "resources": ["/resources/all", "/resources/history", "/resources/processes"],
    "notifications": ["/notifications"],
    "gallery": ["/gallery/albums", "/gallery/stats"],
    "video_station": ["/video/library", "/video/status"],
    "surveillance": ["/surveillance/cameras", "/surveillance/status"],
    "downloads": ["/downloads/list", "/downloads/status"],
    "tickets": ["/tickets/projects"],
    "domains": ["/domains/list", "/domains/ssl"],
    "power": ["/power/schedule", "/power/status"],
    "ups": ["/ups/status"],
    "stickynotes": ["/stickynotes/notes"],
    "rollback": ["/rollback/snapshots"],
    "remote-log": ["/remote-log/sources"],
    "cloud-backup": ["/cloud-backup/status", "/cloud-backup/providers"],
    "naslink": ["/naslink/status", "/naslink/devices"],
    "ssh-manager": ["/ssh/status", "/ssh/keys"],
    "doc-anonymizer": ["/anonymizer/status"],
    "med-assistant": ["/med-assistant/status"],
    "websites": ["/websites/list", "/websites/status"],
    "builder": ["/builder/projects", "/builder/status"],
    "usb-flasher": ["/flasher/devices"],
    "sharing-dlna": ["/dlna/status"],
    "family-hub": ["/familyhub/members"],
    "disk-repair": ["/disk-repair/status"],
    "raid-lvm": ["/raid/status", "/lvm/status"],
    "security-advisor": ["/security/scan/status", "/security/scan/results"],
    "radio-music": ["/radio/stations", "/radio/status"],
}

SECURITY_HEADERS = [
    "X-Content-Type-Options",
    "X-Frame-Options",
    "Strict-Transport-Security",
    "Content-Security-Policy",
    "Referrer-Policy",
    "Permissions-Policy",
]


@dataclass
class QAFinding:
    """A single QA finding / bug."""

    category: str   # network, auth, api, ui, security, performance, workflow, event-log
    severity: str   # critical, high, medium, low
    app: str        # which Ethos app
    title: str
    description: str
    endpoint: str = ""
    response_time: float = 0.0
    status_code: int = 0
    screenshot_label: str = ""
    details: dict = field(default_factory=dict)

    @property
    def ticket_labels(self) -> list[str]:
        labels = ["qa-automated", self.category]
        if self.app:
            labels.append(self.app)
        return labels


@dataclass
class QAReport:
    """Aggregated QA results."""

    target: str
    started_at: float = 0.0
    finished_at: float = 0.0
    findings: list[QAFinding] = field(default_factory=list)
    api_tested: int = 0
    api_passed: int = 0
    api_failed: int = 0
    api_errors: int = 0
    apps_tested: int = 0
    apps_with_errors: int = 0
    workflows_run: int = 0
    workflows_failed: int = 0
    event_log_errors: int = 0

    @property
    def duration(self) -> float:
        return self.finished_at - self.started_at

    @property
    def summary(self) -> str:
        sev: dict[str, int] = {}
        for f in self.findings:
            sev[f.severity] = sev.get(f.severity, 0) + 1
        lines = [
            f"=== QA Report: {self.target} ===",
            f"Duration: {self.duration:.1f}s",
            f"API endpoints tested: {self.api_tested} (pass:{self.api_passed} fail:{self.api_failed} err:{self.api_errors})",
            f"Apps UI-tested: {self.apps_tested} (with errors: {self.apps_with_errors})",
            f"Workflows run: {self.workflows_run} (failed: {self.workflows_failed})",
            f"Event log errors: {self.event_log_errors}",
            f"Total findings: {len(self.findings)}",
        ]
        for s in ("critical", "high", "medium", "low"):
            if c := sev.get(s, 0):
                lines.append(f"  {s.upper()}: {c}")
        if self.findings:
            lines.append("\n-- Top Findings --")
            for f in sorted(self.findings, key=lambda x: {"critical":0,"high":1,"medium":2,"low":3}.get(x.severity, 4))[:30]:
                lines.append(f"  [{f.severity.upper()}] [{f.category}] {f.app}: {f.title}")
        return "\n".join(lines)



class QARunner:
    """Comprehensive QA test runner for Ethos OS NAS."""

    def __init__(self, client: EthosClient, project_name: str = "Ethos",
                 on_finding: Any = None, browser_tool: Any = None,
                 vision_tool: Any = None):
        self._client = client
        self._project_name = project_name
        self._on_finding = on_finding
        self._report = QAReport(target=client.base_url)
        self._browser = browser_tool
        self._vision = vision_tool
        self._saved_results: dict[str, Any] = {}

    @property
    def report(self) -> QAReport:
        return self._report

    async def _add_finding(self, finding: QAFinding) -> None:
        self._report.findings.append(finding)
        logger.warning("QA finding: [%s][%s] %s - %s",
                       finding.severity, finding.category, finding.app, finding.title)
        if self._on_finding:
            try:
                await self._on_finding(finding)
            except Exception as e:
                logger.error("Finding callback failed: %s", e)

    # -- Main entry point ----------------------------------------------------

    async def run_all(self, *, skip: set[str] | None = None) -> QAReport:
        """Run all QA test categories."""
        skip = skip or set()
        self._report.started_at = time.time()
        suites = [
            ("network", self.test_network),
            ("auth", self.test_auth),
            ("security", self.test_security_headers),
            ("api", self.test_all_api_endpoints),
            ("performance", self.test_performance),
            ("event_logs", self.test_event_logs),
            ("browser_ui", self.test_browser_all_apps),
            ("workflows", self.test_workflows),
        ]
        for name, func in suites:
            if name in skip:
                logger.info("Skipping QA suite: %s", name)
                continue
            logger.info("Running QA suite: %s", name)
            try:
                await func()
            except Exception as e:
                logger.error("QA suite '%s' crashed: %s", name, e)
                await self._add_finding(QAFinding(
                    category="infrastructure", severity="high", app="qa-runner",
                    title=f"QA suite '{name}' crashed", description=str(e)))
        self._report.finished_at = time.time()
        return self._report

    # -- Event log monitoring ------------------------------------------------

    async def test_event_logs(self) -> None:
        """Check NAS notification/event log for recent errors."""
        if not self._client.is_authenticated:
            await self._client.login()
        data, err = await self._client.api_safe("/notifications", timeout=15)
        if err:
            await self._add_finding(QAFinding(
                category="event-log", severity="medium", app="system",
                title="Cannot read event log",
                description=f"Failed to fetch /api/notifications: {err}",
                endpoint="/notifications"))
            return
        if not isinstance(data, list):
            return

        cutoff = time.time() - 86400
        recent_errors = [n for n in data
                         if n.get("type") == "error" and n.get("time", 0) > cutoff]
        recent_warnings = [n for n in data
                           if n.get("type") == "warning" and n.get("time", 0) > cutoff]
        self._report.event_log_errors = len(recent_errors)

        error_groups: dict[str, list[dict]] = {}
        for e in recent_errors:
            error_groups.setdefault(e.get("category", "unknown"), []).append(e)

        for category, errors in error_groups.items():
            samples = [e.get("message", "?")[:200] for e in errors[:5]]
            severity = "high" if len(errors) >= 5 else "medium"
            await self._add_finding(QAFinding(
                category="event-log", severity=severity, app=category,
                title=f"{len(errors)} error(s) in event log [{category}]",
                description=(
                    f"{len(errors)} error(s) in last 24h for \'{category}\'."
                    "\nSamples:\n" +
                    "\n".join(f"  * {m}" for m in samples)),
                details={"error_count": len(errors), "samples": samples}))

        if recent_warnings:
            cats = ", ".join(set(w.get("category", "?") for w in recent_warnings))
            await self._add_finding(QAFinding(
                category="event-log", severity="low", app="system",
                title=f"{len(recent_warnings)} warning(s) in event log",
                description=f"{len(recent_warnings)} warning(s) in last 24h. Categories: {cats}"))

        unread = [n for n in data if not n.get("read", True)]
        if len(unread) > 10:
            await self._add_finding(QAFinding(
                category="event-log", severity="low", app="system",
                title=f"{len(unread)} unread notifications",
                description="Large number of unread notifications may indicate unresolved issues."))

    async def _get_event_log_snapshot(self) -> set[int]:
        data, _ = await self._client.api_safe("/notifications", timeout=10)
        if isinstance(data, list):
            return {n.get("id", 0) for n in data}
        return set()

    async def _check_new_events(self, before: set[int], app_name: str) -> None:
        data, _ = await self._client.api_safe("/notifications", timeout=10)
        if not isinstance(data, list):
            return
        for n in data:
            if n.get("id", 0) not in before and n.get("type") == "error":
                await self._add_finding(QAFinding(
                    category="event-log", severity="high", app=app_name,
                    title=f"New error during testing: {n.get('title', '?')}",
                    description=n.get("message", "No details")[:500],
                    details={"event": n}))

    # -- Browser UI testing --------------------------------------------------

    async def test_browser_all_apps(self) -> None:
        """Open every Ethos app in the browser, screenshot, analyze with AI."""
        if not self._browser:
            logger.info("Browser tool not available - skipping UI tests")
            return
        result = await self._browser.execute(action="launch")
        if not result.success:
            await self._add_finding(QAFinding(
                category="ui", severity="high", app="browser",
                title="Browser launch failed",
                description=result.error or "Unknown"))
            return
        try:
            await self._browser_login()
            app_list = await self._get_app_list_from_browser()
            if not app_list:
                await self._add_finding(QAFinding(
                    category="ui", severity="high", app="desktop",
                    title="Cannot discover apps",
                    description="NAS.apps empty or login failed"))
                return
            logger.info("Testing %d apps via browser UI", len(app_list))
            for app_info in app_list:
                app_id = app_info.get("id", "unknown")
                app_name = app_info.get("name", app_id)
                self._report.apps_tested += 1
                events_before = await self._get_event_log_snapshot()
                try:
                    if await self._test_single_app_ui(app_id, app_name):
                        self._report.apps_with_errors += 1
                except Exception as e:
                    self._report.apps_with_errors += 1
                    await self._add_finding(QAFinding(
                        category="ui", severity="medium", app=app_id,
                        title=f"UI test crashed for {app_name}",
                        description=f"{type(e).__name__}: {e}"))
                await self._check_new_events(events_before, app_id)
                # Close the app window
                await self._browser.execute(
                    action="evaluate",
                    expression=(
                        "(function(){var b=document.querySelectorAll("
                        "'.window-close,.close-btn');"
                        "if(b.length){b[b.length-1].click();return 'closed';}"
                        "return 'none';})()"))
                await asyncio.sleep(0.5)
        finally:
            await self._browser.close()

    async def _browser_login(self) -> None:
        await self._browser.execute(
            action="navigate", url=self._client.base_url)
        await asyncio.sleep(2)
        result = await self._browser.execute(
            action="get_elements",
            selector="#loginForm, .login-form, [class*=login]")
        if result.artifacts and result.artifacts.get("count", 0) > 0:
            user = self._client._username
            pw = self._client._password
            js_login = (
                "(function(){"
                "var u=document.querySelector("
                "'#username,input[name=username],input[type=text]');"
                "var p=document.querySelector("
                "'#password,input[name=password],input[type=password]');"
                "var b=document.querySelector("
                "'#loginBtn,button[type=submit],.login-btn');"
                "if(u)u.value='" + user + "';"
                "if(p)p.value='" + pw + "';"
                "if(u)u.dispatchEvent(new Event('input',{bubbles:true}));"
                "if(p)p.dispatchEvent(new Event('input',{bubbles:true}));"
                "if(b){setTimeout(function(){b.click()},200);"
                "return 'submitted';}return 'no btn';})()")
            await self._browser.execute(
                action="evaluate", expression=js_login)
            await asyncio.sleep(3)
            await self._browser.execute(
                action="wait_for",
                selector=".desktop, .taskbar, #desktop, [class*=desktop]",
                timeout=15000)
            await asyncio.sleep(1)
        else:
            logger.info("No login form - may already be logged in")

    async def _get_app_list_from_browser(self) -> list[dict]:
        result = await self._browser.execute(
            action="evaluate",
            expression=(
                "JSON.stringify(typeof NAS!=='undefined'"
                "&&NAS.apps?NAS.apps:[])"))
        if result.success and result.output and result.output != "[]":
            try:
                return _json.loads(result.output)
            except (ValueError, TypeError):
                pass
        data, _ = await self._client.api_safe("/apps", timeout=10)
        return data if isinstance(data, list) else []

    async def _test_single_app_ui(self, app_id: str, app_name: str) -> bool:
        """Test one app: open, screenshot, vision-analyze, check errors.

        Returns True if errors found.
        """
        had_errors = False
        logger.info("UI testing: %s (%s)", app_name, app_id)

        # Clear error buffers
        await self._browser.execute(action="get_console_errors")
        await self._browser.execute(action="get_network_errors")

        # Open app via NAS.apps
        open_js = (
            "(function(){"
            "var a=typeof NAS!=='undefined'"
            "&&NAS.apps&&NAS.apps.find("
            "function(x){return x.id==='" + app_id + "'});"
            "if(a){openApp(a);return 'opened '+a.name;}"
            "return 'not found';})()")
        r = await self._browser.execute(
            action="evaluate", expression=open_js)
        if "not found" in (r.output or ""):
            await self._add_finding(QAFinding(
                category="ui", severity="medium", app=app_id,
                title=f"App '{app_name}' not found in NAS.apps",
                description="App registered but not found at runtime"))
            return True

        await asyncio.sleep(3)

        # Screenshot + Vision analysis
        screenshot = await self._browser.execute(action="screenshot")
        b64 = ""
        if screenshot.success and screenshot.artifacts:
            b64 = screenshot.artifacts.get("screenshot", "")
        if self._vision and b64:
            try:
                vr = await self._vision.execute(
                    action="analyze", image_base64=b64,
                    prompt=(
                        f"Analyze '{app_name}' app screenshot as QA engineer. "
                        "Report: 1) Did it load correctly? 2) Visual bugs? "
                        "3) Error messages? 4) Missing elements? "
                        "5) Rate: OK/Minor/Major/Broken"))
                if vr.success:
                    text = (vr.output or "").lower()
                    bad_words = [
                        "major", "broken", "error message",
                        "failed to load", "blank screen", "crash"]
                    if any(k in text for k in bad_words):
                        sev = "high" if ("broken" in text or "crash" in text) else "medium"
                        await self._add_finding(QAFinding(
                            category="ui", severity=sev, app=app_id,
                            title=f"Vision AI found issues in {app_name}",
                            description=(vr.output or "")[:1000],
                            screenshot_label=f"{app_id}_main"))
                        had_errors = True
            except Exception as e:
                logger.debug("Vision failed for %s: %s", app_id, e)

        # Console errors
        cr = await self._browser.execute(action="get_console_errors")
        if cr.success and "No console errors" not in (cr.output or ""):
            errs = cr.output or ""
            ec = errs.count("[error]")
            wc = errs.count("[warning]")
            if ec > 0:
                had_errors = True
                await self._add_finding(QAFinding(
                    category="ui",
                    severity="high" if ec >= 3 else "medium",
                    app=app_id,
                    title=f"{ec} JS console error(s) in {app_name}",
                    description=errs[:1000],
                    screenshot_label=f"{app_id}_console"))
            if wc > 3:
                await self._add_finding(QAFinding(
                    category="ui", severity="low", app=app_id,
                    title=f"{wc} JS warnings in {app_name}",
                    description=errs[:500]))

        # Network errors
        nr = await self._browser.execute(action="get_network_errors")
        if nr.success and "No network errors" not in (nr.output or ""):
            net = nr.output or ""
            if " 5" in net:
                had_errors = True
                await self._add_finding(QAFinding(
                    category="ui", severity="high", app=app_id,
                    title=f"Server errors (5xx) loading {app_name}",
                    description=net[:1000]))
            elif "failed request" in net.lower():
                await self._add_finding(QAFinding(
                    category="ui", severity="low", app=app_id,
                    title=f"Network errors loading {app_name}",
                    description=net[:500]))

        # Page text error check
        tr = await self._browser.execute(action="get_text")
        pt = (tr.output or "").lower()
        error_texts = [
            "internal server error", "traceback",
            "unhandled exception", "syntax error"]
        if any(e in pt for e in error_texts):
            had_errors = True
            await self._add_finding(QAFinding(
                category="ui", severity="high", app=app_id,
                title=f"Error text visible in {app_name}",
                description=f"Page contains error text. Excerpt: {pt[:300]}"))
        return had_errors

    # -- Workflow testing -----------------------------------------------------

    async def test_workflows(self) -> None:
        """Run detailed workflow tests for apps with definitions."""
        if not self._browser:
            logger.info("Browser not available - skipping workflows")
            return
        result = await self._browser.execute(action="launch")
        if not result.success and "already running" not in (result.output or ""):
            return
        try:
            await self._browser_login()
            for app_id, workflow in ETHOS_APP_WORKFLOWS.items():
                wf_name = workflow["name"]
                self._report.workflows_run += 1
                self._saved_results.clear()
                logger.info("Workflow: %s (%s)", wf_name, app_id)
                events_before = await self._get_event_log_snapshot()

                # Open app
                open_js = (
                    "(function(){"
                    "var a=typeof NAS!=='undefined'"
                    "&&NAS.apps&&NAS.apps.find("
                    "function(x){return x.id==='" + app_id + "'});"
                    "if(a){openApp(a);return 'ok';}return 'nf';})()")
                await self._browser.execute(
                    action="evaluate", expression=open_js)
                await asyncio.sleep(2)

                wf_failed = False
                cleanup_steps = []
                for step in workflow.get("steps", []):
                    try:
                        if step.get("is_cleanup"):
                            cleanup_steps.append(step)
                            continue
                        if not await self._exec_step(step, app_id, wf_name):
                            wf_failed = True
                    except Exception as e:
                        wf_failed = True
                        await self._add_finding(QAFinding(
                            category="workflow", severity="medium", app=app_id,
                            title=(
                                f"Step failed: "
                                f"{step.get('label', step.get('action', '?'))}"),
                            description=f"{type(e).__name__}: {e}"))

                for step in cleanup_steps:
                    try:
                        await self._exec_step(step, app_id, wf_name)
                    except Exception:
                        pass

                if wf_failed:
                    self._report.workflows_failed += 1
                await self._check_new_events(events_before, app_id)

                # Close window
                await self._browser.execute(
                    action="evaluate",
                    expression=(
                        "(function(){var b=document.querySelectorAll("
                        "'.window-close,.close-btn');"
                        "if(b.length){b[b.length-1].click();}"
                        "return 'ok';})()"))
                await asyncio.sleep(0.5)
        finally:
            await self._browser.close()

    async def _exec_step(self, step: dict, app_id: str, wf_name: str) -> bool:
        action = step.get("action", "")

        if action == "wait":
            r = await self._browser.execute(
                action="wait_for",
                selector=step["selector"],
                timeout=step.get("timeout", 8000))
            if not r.success:
                await self._add_finding(QAFinding(
                    category="workflow", severity="medium", app=app_id,
                    title=f"{wf_name}: element not found",
                    description=f"Timeout for: {step['selector']}"))
                return False

        elif action == "screenshot":
            await self._browser.execute(action="screenshot")

        elif action == "check_errors":
            cr = await self._browser.execute(action="get_console_errors")
            if cr.success and "[error]" in (cr.output or ""):
                c = (cr.output or "").count("[error]")
                await self._add_finding(QAFinding(
                    category="workflow", severity="medium", app=app_id,
                    title=f"{wf_name}: {c} JS error(s)",
                    description=(cr.output or "")[:800]))
            nr = await self._browser.execute(action="get_network_errors")
            if nr.success and " 5" in (nr.output or ""):
                await self._add_finding(QAFinding(
                    category="workflow", severity="high", app=app_id,
                    title=f"{wf_name}: server error",
                    description=(nr.output or "")[:800]))
                return False

        elif action == "evaluate":
            await self._browser.execute(
                action="evaluate", expression=step["js"])

        elif action == "sleep":
            await asyncio.sleep(step.get("ms", 1000) / 1000)

        elif action == "assert_elements":
            r = await self._browser.execute(
                action="get_elements", selector=step["selector"])
            cnt = r.artifacts.get("count", 0) if r.artifacts else 0
            min_c = step.get("min_count", 1)
            if cnt < min_c:
                await self._add_finding(QAFinding(
                    category="workflow", severity="medium", app=app_id,
                    title=f"{wf_name}: expected >={min_c} elements, got {cnt}",
                    description=step.get("description", step["selector"])))
                return False

        elif action == "assert_text":
            r = await self._browser.execute(action="get_text")
            if step["text"].lower() not in (r.output or "").lower():
                await self._add_finding(QAFinding(
                    category="workflow", severity="medium", app=app_id,
                    title=f"{wf_name}: text not found",
                    description=step.get(
                        "description",
                        f"Expected '{step['text']}'") ))
                return False

        elif action == "api_call":
            method = step.get("method", "GET")
            path = step["path"]
            body = step.get("body")
            if method == "GET":
                data, err = await self._client.api_safe(
                    path, timeout=15)
            else:
                data, err = await self._client.api_safe(
                    path, method=method, body=body, timeout=15)
            if err:
                sev = "low" if "404" in err or "405" in err else "medium"
                await self._add_finding(QAFinding(
                    category="workflow", severity=sev, app=app_id,
                    title=(
                        f"{wf_name}: API failed - "
                        f"{step.get('label', path)}"),
                    description=f"{method} /api{path}: {err[:500]}",
                    endpoint=path))
                return False
            if step.get("save_result") and isinstance(data, dict):
                self._saved_results[step["save_result"]] = data
            if (step.get("expect_contains")
                    and step["expect_contains"] not in str(data)):
                await self._add_finding(QAFinding(
                    category="workflow", severity="medium", app=app_id,
                    title=(
                        f"{wf_name}: unexpected response - "
                        f"{step.get('label', path)}"),
                    description=(
                        f"Expected '{step['expect_contains']}' in "
                        f"{method} /api{path}"),
                    endpoint=path))
                return False

        elif action == "api_call_dynamic":
            method = step.get("method", "GET")
            path = self._resolve_template(step["path_template"])
            if not path:
                return False
            body = step.get("body")
            if method in ("GET", "DELETE"):
                data, err = await self._client.api_safe(
                    path, method=method, timeout=15)
            else:
                data, err = await self._client.api_safe(
                    path, method=method, body=body, timeout=15)
            if err and not step.get("is_cleanup"):
                await self._add_finding(QAFinding(
                    category="workflow", severity="medium", app=app_id,
                    title=(
                        f"{wf_name}: dynamic API failed - "
                        f"{step.get('label', path)}"),
                    description=f"{method} /api{path}: {err[:500]}",
                    endpoint=path))
                return False

        return True

    def _resolve_template(self, template: str) -> str | None:
        def repl(m):
            key, fld = m.group(1), m.group(2)
            obj = self._saved_results.get(key)
            if obj and isinstance(obj, dict):
                return str(obj.get(fld, obj.get("id", "")))
            return ""
        result = _re.sub(r'\{(\w+)\.(\w+)\}', repl, template)
        return result if result and "{" not in result else None

    # -- Network tests -------------------------------------------------------

    async def test_network(self) -> None:
        import socket
        import ssl
        from urllib.parse import urlparse

        parsed = urlparse(self._client.base_url)
        host = parsed.hostname or ""
        port = parsed.port or (443 if parsed.scheme == "https" else 80)

        try:
            socket.getaddrinfo(host, None)
        except socket.gaierror as e:
            await self._add_finding(QAFinding(
                category="network", severity="critical", app="dns",
                title=f"DNS failed for {host}", description=str(e)))
            return

        try:
            r, w = await asyncio.wait_for(
                asyncio.open_connection(host, port), timeout=10)
            w.close()
            await w.wait_closed()
        except Exception as e:
            await self._add_finding(QAFinding(
                category="network", severity="critical", app="tcp",
                title=f"Cannot connect to {host}:{port}",
                description=str(e)))
            return

        if parsed.scheme == "https":
            try:
                ctx = ssl.create_default_context()
                r, w = await asyncio.wait_for(
                    asyncio.open_connection(host, port, ssl=ctx), timeout=10)
                w.close()
                await w.wait_closed()
            except ssl.SSLCertVerificationError as e:
                await self._add_finding(QAFinding(
                    category="network", severity="medium", app="ssl",
                    title="SSL cert validation failed",
                    description=str(e)))
            except Exception:
                pass

        ports_to_scan = {
            22: "SSH", 80: "HTTP", 443: "HTTPS", 445: "SMB",
            139: "NetBIOS", 548: "AFP", 2049: "NFS", 8080: "HTTP-Alt",
            9090: "Webmin", 5001: "DLNA", 8096: "Jellyfin",
            51820: "WireGuard",
        }
        open_p = []
        for p, svc in ports_to_scan.items():
            try:
                r, w = await asyncio.wait_for(
                    asyncio.open_connection(host, p), timeout=2)
                w.close()
                await w.wait_closed()
                open_p.append(f"{p}/{svc}")
            except Exception:
                pass
        logger.info("Open ports: %s", ", ".join(open_p))

    # -- Auth tests ----------------------------------------------------------

    async def test_auth(self) -> None:
        async with httpx.AsyncClient(verify=False, timeout=15) as http:
            base = self._client.base_url

            # Invalid creds
            try:
                resp = await http.post(
                    f"{base}/api/auth/login",
                    json={"username": "invalid_xyz", "password": "wrong"})
                if resp.status_code == 200 and resp.json().get("token"):
                    await self._add_finding(QAFinding(
                        category="auth", severity="critical", app="auth",
                        title="Invalid credentials accepted",
                        description="Token issued for invalid user",
                        status_code=resp.status_code))
            except Exception:
                pass

            # No token
            try:
                resp = await http.get(f"{base}/api/dashboard/summary")
                if resp.status_code == 200:
                    await self._add_finding(QAFinding(
                        category="auth", severity="critical", app="auth",
                        title="API accessible without auth",
                        description="/api/dashboard/summary 200 with no token",
                        endpoint="/dashboard/summary", status_code=200))
            except Exception:
                pass

            # Garbage token
            try:
                resp = await http.get(
                    f"{base}/api/dashboard/summary",
                    headers={"Authorization": "Bearer garbage_token_12345"})
                if resp.status_code == 200:
                    await self._add_finding(QAFinding(
                        category="auth", severity="critical", app="auth",
                        title="API accepts invalid token",
                        description="200 with garbage bearer token",
                        endpoint="/dashboard/summary", status_code=200))
            except Exception:
                pass

            # SQL injection
            try:
                resp = await http.post(
                    f"{base}/api/auth/login",
                    json={
                        "username": "' OR 1=1 --",
                        "password": "' OR 1=1 --"})
                if resp.status_code == 200 and resp.json().get("token"):
                    await self._add_finding(QAFinding(
                        category="auth", severity="critical", app="auth",
                        title="SQL injection accepted in login",
                        description="Token issued for SQLi payload",
                        status_code=resp.status_code))
            except Exception:
                pass

            # XSS reflection
            try:
                resp = await http.post(
                    f"{base}/api/auth/login",
                    json={
                        "username": "<script>alert(1)</script>",
                        "password": "x"})
                if "<script>alert(1)</script>" in resp.text:
                    await self._add_finding(QAFinding(
                        category="auth", severity="high", app="auth",
                        title="XSS reflected in login",
                        description="Script tag echoed unescaped"))
            except Exception:
                pass

            # Rate limiting check
            try:
                statuses = []
                for _ in range(10):
                    r = await http.post(
                        f"{base}/api/auth/login",
                        json={"username": "admin", "password": "wrong"})
                    statuses.append(r.status_code)
                if all(s not in (429, 403) for s in statuses):
                    await self._add_finding(QAFinding(
                        category="auth", severity="medium", app="auth",
                        title="No rate limiting on login",
                        description=(
                            f"10 rapid fails, status codes: "
                            f"{set(statuses)}")))
            except Exception:
                pass

    # -- Security headers ----------------------------------------------------

    async def test_security_headers(self) -> None:
        async with httpx.AsyncClient(verify=False, timeout=15) as http:
            try:
                resp = await http.get(self._client.base_url)
            except Exception as e:
                await self._add_finding(QAFinding(
                    category="security", severity="medium", app="web",
                    title="Cannot fetch main page", description=str(e)))
                return

            resp_hdrs = {k.lower() for k in resp.headers.keys()}
            missing = [h for h in SECURITY_HEADERS
                       if h.lower() not in resp_hdrs]
            if missing:
                await self._add_finding(QAFinding(
                    category="security", severity="medium", app="web",
                    title=f"Missing security headers: {', '.join(missing)}",
                    description=f"Missing: {', '.join(missing)}",
                    endpoint="/", status_code=resp.status_code,
                    details={"missing_headers": missing}))

            for cookie in resp.headers.get_list("set-cookie"):
                issues = []
                cl = cookie.lower()
                if "httponly" not in cl:
                    issues.append("HttpOnly")
                if "secure" not in cl:
                    issues.append("Secure")
                if "samesite" not in cl:
                    issues.append("SameSite")
                if issues:
                    name = cookie.split("=")[0].strip()
                    await self._add_finding(QAFinding(
                        category="security", severity="medium", app="web",
                        title=(
                            f"Cookie '{name}' missing: "
                            f"{', '.join(issues)}"),
                        description=f"Cookie: {cookie[:120]}"))

            server = resp.headers.get("server", "")
            version_hints = ["nginx/", "apache/", "python/"]
            if server and any(v in server.lower() for v in version_hints):
                await self._add_finding(QAFinding(
                    category="security", severity="low", app="web",
                    title=f"Server version disclosed: {server}",
                    description="Server header reveals version info."))

    # -- API endpoint tests --------------------------------------------------

    async def test_all_api_endpoints(self) -> None:
        if not self._client.is_authenticated:
            await self._client.login()
        all_ep = [
            (app, p)
            for app, paths in ETHOS_APP_ENDPOINTS.items()
            for p in paths
        ]
        sem = asyncio.Semaphore(5)

        async def test_one(app: str, path: str) -> None:
            async with sem:
                self._report.api_tested += 1
                start = time.time()
                try:
                    data, err = await self._client.api_safe(
                        path, timeout=20)
                    elapsed = time.time() - start
                    if err:
                        self._report.api_errors += 1
                        if "404" not in err and "405" not in err:
                            await self._add_finding(QAFinding(
                                category="api", severity="medium", app=app,
                                title=f"API error: {path}",
                                description=err[:500],
                                endpoint=path,
                                response_time=elapsed))
                    else:
                        self._report.api_passed += 1
                        if isinstance(data, dict) and data.get("error"):
                            self._report.api_failed += 1
                            self._report.api_passed -= 1
                            await self._add_finding(QAFinding(
                                category="api", severity="low", app=app,
                                title=f"API returned error: {path}",
                                description=str(data["error"])[:500],
                                endpoint=path,
                                response_time=elapsed))
                except Exception as e:
                    self._report.api_errors += 1
                    await self._add_finding(QAFinding(
                        category="api", severity="medium", app=app,
                        title=f"API exception: {path}",
                        description=f"{type(e).__name__}: {e}",
                        endpoint=path))

        await asyncio.gather(*[test_one(app, p) for app, p in all_ep])

    # -- Performance tests ---------------------------------------------------

    async def test_performance(self) -> None:
        threshold = 5.0
        endpoints = [
            ("/dashboard/summary", "dashboard"),
            ("/files/list?path=/home", "files"),
            ("/storage/drives", "storage"),
            ("/services/list", "services"),
            ("/resources/all", "resources"),
            ("/docker/containers", "docker"),
            ("/users/list", "users"),
            ("/apps", "apps"),
        ]
        for path, app in endpoints:
            start = time.time()
            await self._client.api_safe(path, timeout=30)
            elapsed = time.time() - start
            if elapsed > threshold:
                await self._add_finding(QAFinding(
                    category="performance",
                    severity="medium" if elapsed < 10 else "high",
                    app=app,
                    title=f"Slow: {path} ({elapsed:.1f}s)",
                    description=(
                        f"{elapsed:.1f}s exceeds "
                        f"{threshold}s threshold"),
                    endpoint=path,
                    response_time=elapsed))

    # -- Ticket creation -----------------------------------------------------

    async def create_tickets(self, project_name: str = "Ethos") -> list[dict]:
        from codator.infrastructure.tools.ethos_ticket_tool import (
            EthosTicketTool,
        )
        tool = EthosTicketTool(client=self._client)
        try:
            await tool._resolve_project_id(project_name)
        except ValueError:
            logger.error("Project '%s' not found", project_name)
            return []

        created = []
        for f in self._report.findings:
            r = await tool.execute(
                action="create", project=project_name,
                title=f"[QA-{f.severity.upper()}] {f.title}",
                description=self._format_finding_body(f),
                type="bug" if f.category != "performance" else "task",
                priority=f.severity, column="Backlog",
                labels=f.ticket_labels)
            if r.success:
                created.append(r.artifacts)
                logger.info("Created ticket: %s", f.title)
            else:
                logger.error("Ticket failed: %s - %s", f.title, r.error)
        return created

    @staticmethod
    def _format_finding_body(finding: QAFinding) -> str:
        lines = [
            f"## {finding.title}", "",
            f"**Category:** {finding.category}",
            f"**Severity:** {finding.severity}",
            f"**App:** {finding.app}",
        ]
        if finding.endpoint:
            lines.append(f"**Endpoint:** `{finding.endpoint}`")
        if finding.status_code:
            lines.append(f"**Status Code:** {finding.status_code}")
        if finding.response_time:
            lines.append(f"**Response Time:** {finding.response_time:.2f}s")
        if finding.screenshot_label:
            lines.append(f"**Screenshot:** {finding.screenshot_label}")
        lines.extend(["", "### Description", "", finding.description])
        if finding.details:
            lines.extend([
                "", "### Details", "",
                f"```json\n{finding.details}\n```"])
        lines.extend([
            "", "---",
            "*Created by codator QA Runner (automated)*"])
        return "\n".join(lines)
