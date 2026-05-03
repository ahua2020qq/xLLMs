<!--
  nxtLLM — Next-Generation LLM Inference Engine
  Copyright (c) 2026 Shanye (山野小娃) <ahua2020@qq.com>
  SPDX-License-Identifier: Apache-2.0
  This header must not be removed. All derivative works must retain this notice.
-->

# Security Policy

## Reporting a Vulnerability

**Do not open a public issue.** To report a security vulnerability, please
create a **private GitHub Security Advisory** or follow these steps:

1. Go to the [Security Advisories](https://github.com/ahua2020qq/nxtLLM/security/advisories) page.
2. Click **"New draft security advisory"**.
3. Fill in the details:
   - **Title**: A brief, clear summary of the vulnerability.
   - **Affected versions**: Which release(s) or commit range are affected.
   - **Description**: Steps to reproduce, potential impact, and any known
     mitigations.
   - **Severity**: Use the CVSS calculator to estimate severity (Low / Medium /
     High / Critical).
4. Submit the advisory.

If you are unable to use GitHub Security Advisories, you may report the issue
via a **private GitHub Issue** labeled `security` — our maintainers will convert
it to a private advisory.

## What to Expect

- **Acknowledgment**: Within 48 hours, a maintainer will acknowledge receipt.
- **Assessment**: We will triage and validate the report, typically within 5
  business days.
- **Fix timeline**: Critical vulnerabilities will be patched within 7 days;
  lower-severity issues are addressed in the next scheduled release.
- **Disclosure**: We follow coordinated disclosure. A CVE will be requested if
  warranted. Public disclosure will be coordinated with the reporter.

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Scope

The following are in scope for the security policy:

- Memory safety bugs (buffer overflows, use-after-free, double-free).
- Input validation issues that may lead to crashes or undefined behavior.
- Concurrency bugs with security implications (data races in the page pool).
- Integer overflow/underflow in allocation or eviction logic.

Out of scope:

- Denial-of-service via resource exhaustion (unless caused by a logic bug).
- Issues in dependencies that are already publicly known.

## Credits

We will acknowledge reporters who follow responsible disclosure, unless they
request anonymity.
