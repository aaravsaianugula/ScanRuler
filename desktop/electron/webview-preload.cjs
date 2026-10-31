// SPDX-License-Identifier: AGPL-3.0-only
// The live view's page: answer ScanRuler's first-run questions before its scripts run, as the e2e
// harness does (scripts/e2e-lib.mjs) — an unanswered agent-data warning would stall the pairing.
// The support card is ScanRuler's own thank-you, switched off here only because it would cover the live
// view the app exists to show; it stays on in ScanRuler itself.
for (const [k, v] of [['scanruler.askstlunits', '0'], ['scanruler.beta', 'all'], ['scanruler.supportask', '0'], ['scanruler.supportcard', '0'], ['scanruler.agentdataok', '1']]) {
  try { localStorage.setItem(k, v) } catch { /* storage blocked: the question is answered by hand */ }
}
