import type { PatientWithSummary, Priority } from './types'

export const PRIORITY_ORDER: Priority[] = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'OK']

// Dashboard order: priority level first, then knowledge-graph risk score
export function byUrgency(a: PatientWithSummary, b: PatientWithSummary): number {
  const pa = a.latest_summary?.priority ?? 'OK'
  const pb = b.latest_summary?.priority ?? 'OK'
  return PRIORITY_ORDER.indexOf(pa) - PRIORITY_ORDER.indexOf(pb)
    || (b.latest_summary?.risk_score ?? 0) - (a.latest_summary?.risk_score ?? 0)
}
