---
title: Information Security Policy
department: Security
type: policy
version: 5.0
effective: 2026-03-01
owner: Office of the CISO
---

# Information Security Policy

## Scope

This policy applies to every employee, contractor and third party with access to
Northwind Systems systems or data. Compliance is a condition of continued
access. Breaches may result in disciplinary action up to dismissal.

## Account security

- Passwords must be at least **14 characters**, unique per service, and stored in
  the company password manager. Never reuse a personal password.
- **Multi-factor authentication is mandatory** on every corporate system. Hardware
  security keys are required for administrators and anyone with production access.
- Never share credentials, including with IT. No member of staff will ever ask
  you for your password.
- Accounts are locked automatically after 30 days of inactivity.

## Device standard

Company-issued laptops are the only devices permitted to hold company data. Each
must have full-disk encryption, the managed endpoint agent, automatic screen lock
after 5 minutes, and operating system updates applied within **14 days of
release** (72 hours for a critical vulnerability).

Jailbroken or rooted devices are prohibited. Do not install software from outside
the approved catalogue; request additions through the IT service desk.

## Data classification

| Class | Examples | Handling |
| --- | --- | --- |
| Public | Marketing site, published docs | No restriction |
| Internal | Team plans, internal wikis | Employees only, do not post externally |
| Confidential | Customer data, contracts, source code | Need-to-know, encrypted in transit and at rest |
| Restricted | Credentials, security keys, personal data of customers | Named individuals only, access logged and reviewed quarterly |

Never move Confidential or Restricted data to a personal account, personal cloud
storage or an unapproved AI service. Approved AI tooling is listed in the IT
service catalogue.

## Access management

Access follows least privilege. Requests are raised through the IT service desk
and require the approval of the data owner. **Privileged and production access
additionally requires Security team approval** and is granted for a fixed period,
not permanently.

Access is reviewed quarterly. Managers must confirm within 10 working days that
their reports still require each entitlement; unconfirmed access is revoked
automatically.

## Remote and network access

Use the corporate VPN on any network you do not control, including hotel and
airport Wi-Fi. Split tunnelling is disabled by design. Do not connect company
devices to unmanaged USB accessories or public charging ports.

## Phishing and social engineering

Report suspicious messages with the "Report Phishing" button rather than
deleting them. Finance requests received by email or chat that ask for a payment,
a change of bank details or a gift-card purchase must be verified by voice call
to a known number before any action.

## Incident reporting

Report any suspected security incident - lost device, mistaken disclosure,
malware alert, unexpected password reset - to the Security team **within 24
hours** through the IT service desk at critical priority, or by calling the
security hotline out of hours.

Do not attempt to investigate or remediate an incident yourself; preserve the
device state and evidence. There is no penalty for reporting in good faith,
including for an incident you caused.

## Secure development

Source code lives in the corporate repositories only. Secrets are never
committed; use the secrets manager. Every change requires review by another
engineer, and production deployments require an approved change record.
Dependencies with known critical vulnerabilities must be patched within 7 days.

## Physical security

Wear your badge on site, do not hold doors open for people you do not recognise,
and clear desks of Confidential material overnight. Visitors must be signed in
and escorted.

## Leaving the company

All equipment must be returned on or before the last working day. Access is
revoked at the end of the final shift. Retaining company data after departure is
a criminal offence in several jurisdictions in which the company operates.
