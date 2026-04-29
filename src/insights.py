# src/insights.py
import logging
import numpy as np
import pandas as pd
from typing import Any
from src.metrics import classify_severity

log = logging.getLogger("insights")

SEVERITY_ORDER = {"critical": 0, "warning": 1, "positive": 2, "info": 3, "neutral": 4}
InsightCard = dict[str, str]

# This module defines a set of rules that analyze key performance indicators (KPIs) related to the UAC system and generate 
# insight cards with severity classifications and actionable recommendations. Each rule function evaluates specific KPIs against 
# predefined thresholds to determine if the situation is critical, warning, or positive, and constructs a corresponding insight card. 
# The main function `generate_insights` applies all rules to the current KPIs and compiles a sorted list of insights for display in the dashboard.

def _rule_system_load_level(kpis, df):
    val = kpis.get("total_under_care", 0)
    sev = classify_severity("total_under_care", val)
    if sev == "critical":
        return {"title": f"Critical System Load: {val:,.0f} Children",
                "body": f"Combined CBP+HHS population of {val:,.0f} exceeds the critical threshold of 18,000. Shelter overcrowding and medical capacity strain are statistically likely.",
                "severity": "critical", "recommendation": "Activate emergency shelter expansion. Coordinate with FEMA and NGO partners."}
    if sev == "warning":
        return {"title": f"System Load Elevated: {val:,.0f} Children",
                "body": f"Total population of {val:,.0f} is above the 12,000 warning threshold. Proactive capacity management is advised.",
                "severity": "warning", "recommendation": "Review shelter utilization. Begin 20-30% surge contingency planning."}
    return {"title": f"System Load Within Managed Range: {val:,.0f}",
            "body": f"Total population of {val:,.0f} is below alert thresholds. Capacity appears adequate.",
            "severity": "positive", "recommendation": "Maintain current discharge and placement pace."}


def _rule_net_intake_pressure(kpis, df):
    val = kpis.get("net_intake_pressure", 0)
    sev = classify_severity("net_intake_pressure", val)
    if val > 0:
        if sev == "critical":
            return {"title": f"Severe Intake Pressure: +{val:,.0f} Children/Day",
                    "body": f"HHS is absorbing {val:,.0f} more children per day than it releases. At this rate the backlog compounds rapidly.",
                    "severity": "critical", "recommendation": "Prioritize expedited case review. Request additional ORR bed authorizations immediately."}
        if sev == "warning":
            return {"title": f"Moderate Intake Pressure: +{val:,.0f} Children/Day",
                    "body": f"System adding {val:,.0f} net children per day. Could accumulate into a significant backlog within weeks.",
                    "severity": "warning", "recommendation": "Accelerate sponsor vetting and increase case manager staffing."}
        return {"title": f"Low Intake Pressure: +{val:,.0f} Children/Day",
                "body": "Net intake is slightly positive, within expected operational range given daily transfer variability.",
                "severity": "info", "recommendation": "Monitor weekly rolling averages to confirm trend stability."}
    return {"title": f"Discharge Surplus: {val:,.0f} Children/Day",
            "body": f"HHS discharged {abs(val):,.0f} more children than it received — effective case processing signal.",
            "severity": "positive", "recommendation": "Maintain current discharge pace. Document effective workflows for replication."}


def _rule_discharge_effectiveness(kpis, df):
    ratio = kpis.get("discharge_offset_ratio", 0)
    base  = kpis.get("hhs_in_care_latest", 0)
    sev   = classify_severity("discharge_offset_ratio", ratio)
    if sev == "critical":
        return {"title": f"Very Low Discharge Rate: {ratio:.2%}",
                "body": f"Only {ratio:.2%} of the HHS population ({base:,.0f}) discharged daily. May indicate processing bottlenecks or legal holds.",
                "severity": "critical", "recommendation": "Root-cause audit on discharge pipeline. Identify legal, administrative, or sponsor bottlenecks."}
    if sev == "warning":
        return {"title": f"Discharge Rate Below Target: {ratio:.2%}",
                "body": f"Discharge effectiveness of {ratio:.2%} against {base:,.0f} children is below warning threshold.",
                "severity": "warning", "recommendation": "Review 7-day rolling discharge rates to distinguish cyclical dips from structural decline."}
    return {"title": f"Discharge Rate Operationally Normal: {ratio:.2%}",
            "body": f"A ratio of {ratio:.2%} against {base:,.0f} children is consistent with expected throughput. This does not indicate system failure.",
            "severity": "positive", "recommendation": "No immediate action required on discharge rate."}


def _rule_backlog_accumulation(kpis, df):
    pct   = kpis.get("backlog_pct", 0)
    days  = kpis.get("backlog_days", 0)
    total = kpis.get("total_days", 1)
    mx    = kpis.get("max_consecutive_backlog", 0)
    sev   = classify_severity("backlog_pct", pct)
    if sev == "critical":
        return {"title": f"Chronic Backlog: {pct:.1f}% of Days Under Pressure",
                "body": f"System under positive pressure on {days}/{total} days ({pct:.1f}%). Longest streak: {mx} days. Indicates structural imbalance.",
                "severity": "critical", "recommendation": "Commission full pipeline throughput audit. Consider policy interventions for expedited release."}
    if sev == "warning":
        return {"title": f"Elevated Backlog Frequency: {pct:.1f}% of Days",
                "body": f"Positive intake pressure on {days}/{total} days, max streak {mx} days. Warrants monitoring.",
                "severity": "warning", "recommendation": "Set 30-day monitoring window. Flag if rate exceeds 70% next month."}
    return {"title": f"Backlog Frequency Normal: {pct:.1f}%",
            "body": f"Intake pressure positive on {days}/{total} days, max streak {mx} days. Within expected operational variance.",
            "severity": "positive", "recommendation": "Continue monitoring. No structural intervention required."}


def _rule_volatility(kpis, df):
    vol = kpis.get("volatility_index", 0)
    sev = classify_severity("volatility_index", vol)
    if sev == "critical":
        return {"title": f"High System Volatility: sigma={vol:,.0f}",
                "body": f"14-day rolling std of {vol:,.0f} above critical threshold of 1,500. Complicates staffing and bed allocation.",
                "severity": "critical", "recommendation": "Implement surge buffer protocols. Pre-position flexible staffing contracts."}
    if sev == "warning":
        return {"title": f"Moderate Volatility: sigma={vol:,.0f}",
                "body": f"Load volatility of {vol:,.0f} above warning threshold. May complicate near-term resource planning.",
                "severity": "warning", "recommendation": "Use 14-day rolling averages rather than daily counts for staffing decisions."}
    return {"title": f"System Load Stable: sigma={vol:,.0f}",
            "body": f"14-day volatility of {vol:,.0f} is within manageable range. Day-to-day swings unlikely to disrupt planning.",
            "severity": "positive", "recommendation": "Maintain current planning cadence."}


def _rule_mom_trend(kpis, df):
    mom = kpis.get("mom_load_change")
    if mom is None or np.isnan(mom):
        return {"title": "Month-over-Month: Insufficient Data",
                "body": "At least 60 days required for MoM comparison. Expand the date range.",
                "severity": "neutral", "recommendation": "Select a wider date range in the sidebar."}
    sev  = classify_severity("mom_load_change", abs(mom)) if mom > 0 else "positive"
    word = "increased" if mom > 0 else "decreased"
    icon = "up" if mom > 0 else "down"
    if mom > 0 and sev in ("critical", "warning"):
        return {"title": f"System Load {word.title()} {mom:+.1f}% MoM",
                "body": f"Average load in last 30 days is {mom:.1f}% higher than preceding 30 days. Requires proactive adjustment.",
                "severity": sev, "recommendation": "Forecast load in the Forecast tab. Begin procurement conversations for additional capacity."}
    return {"title": f"Month-over-Month Load Change: {mom:+.1f}%",
            "body": f"System load has {word} by {abs(mom):.1f}% vs prior 30-day period. Within manageable range.",
            "severity": "info" if mom > 0 else "positive", "recommendation": "Monitor trend weekly to confirm direction."}


def _rule_transfer_pipeline(kpis, df):
    ratio   = kpis.get("transfer_to_intake_ratio", 1.0)
    total_a = kpis.get("total_apprehensions", 0)
    total_t = kpis.get("total_transfers", 0)
    sev     = classify_severity("transfer_to_intake_ratio", ratio)
    if sev == "critical":
        return {"title": f"Transfer Pipeline Bottleneck: {ratio:.1%} Transfer Rate",
                "body": f"Only {ratio:.1%} of {total_a:,.0f} apprehensions resulted in HHS transfer ({total_t:,.0f}). Rate below 50% signals serious handoff bottleneck.",
                "severity": "critical", "recommendation": "Audit CBP transfer processing times. Verify 72-hour Flores limit compliance."}
    if sev == "warning":
        return {"title": f"Transfer Rate Below Benchmark: {ratio:.1%}",
                "body": f"CBP->HHS transfer rate of {ratio:.1%} below 75% warning threshold. May signal processing delays.",
                "severity": "warning", "recommendation": "Cross-reference with CBP custody duration data to identify extended-hold cases."}
    return {"title": f"Transfer Pipeline Healthy: {ratio:.1%}",
            "body": f"{ratio:.1%} of apprehended children transferred to HHS — consistent with expected pipeline throughput.",
            "severity": "positive", "recommendation": "No pipeline intervention required."}


def _rule_cbp_custody_level(kpis, df):
    val = kpis.get("cbp_in_custody_latest", 0)
    sev = classify_severity("cbp_in_custody_latest", val)
    if sev in ("critical", "warning"):
        return {"title": f"High CBP Custody Count: {val:,.0f} Children",
                "body": f"{val:,.0f} children in CBP custody above {'critical (6,000)' if sev=='critical' else 'warning (3,000)'} threshold. CBP facilities not designed for extended stays.",
                "severity": sev, "recommendation": "Accelerate HHS intake capacity. Escalate to legal if 72-hour limits are exceeded."}
    return None


def _rule_consecutive_streak(kpis, df):
    mx  = kpis.get("max_consecutive_backlog", 0)
    sev = classify_severity("max_consecutive_backlog", mx)
    if sev == "critical":
        return {"title": f"Extended Backlog Streak: {mx} Consecutive Days",
                "body": f"A {mx}-day unbroken backlog period was detected. Streaks of this length historically precede shelter emergency declarations.",
                "severity": "critical", "recommendation": "Review the period in Bottleneck Analysis tab. Target discharge or intake intervention based on root cause."}
    if sev == "warning":
        return {"title": f"Notable Backlog Streak: {mx} Consecutive Days",
                "body": f"A {mx}-day consecutive backlog detected — below critical 21-day threshold but warrants investigation.",
                "severity": "warning", "recommendation": "Use Bottleneck Analysis tab to identify period and review external factors."}
    return None


_RULES = [
    _rule_system_load_level,
    _rule_net_intake_pressure,
    _rule_discharge_effectiveness,
    _rule_backlog_accumulation,
    _rule_consecutive_streak,
    _rule_cbp_custody_level,
    _rule_volatility,
    _rule_mom_trend,
    _rule_transfer_pipeline,
]


def generate_insights(kpis: dict[str, Any], df: pd.DataFrame) -> list[InsightCard]:
    cards: list[InsightCard] = []
    for rule_fn in _RULES:
        try:
            card = rule_fn(kpis, df)
            if card is not None:
                card.setdefault("recommendation", "No specific action recommended.")
                cards.append(card)
        except Exception as exc:
            log.error("Rule '%s' failed: %s", rule_fn.__name__, exc, exc_info=True)
            cards.append({"title": f"Insight Rule Error: {rule_fn.__name__}",
                          "body": str(exc), "severity": "neutral",
                          "recommendation": "Check application logs."})
    cards.sort(key=lambda c: SEVERITY_ORDER.get(c.get("severity", "neutral"), 99))
    log.info("Generated %d insight cards.", len(cards))
    return cards
