"""
Sector classification used by both per-ticker labeling (Polygon SIC →
sector enum) and the cross-sectional normalization step in the main
feature_engineer pipeline.

The 12-sector taxonomy is chosen so that stocks within each bucket tend
to respond to the same macro drivers (rates, oil, consumer spending,
etc.). It's coarser than full GICS but more useful than 1-letter SIC
divisions.
"""

from typing import Dict, Tuple


SECTOR_NAMES = [
    "energy",               # 0
    "materials_industrial", # 1
    "tech_hardware",        # 2
    "software_services",    # 3
    "healthcare",           # 4
    "financials",           # 5
    "real_estate",          # 6
    "consumer_disc",        # 7
    "consumer_staples",     # 8
    "telecom_media",        # 9
    "utilities",            # 10
    "other",                # 11
]
N_SECTORS = len(SECTOR_NAMES)


def _sic_to_sector(sic_code: str) -> Tuple[int, str]:
    """
    Map a 4-digit SIC code to one of 12 trading-relevant sectors.

    Groups are chosen so that stocks within each sector tend to respond
    to the same macro drivers (rates, oil, consumer spending, etc.).
    """
    try:
        sic = int(str(sic_code)[:2])
    except (ValueError, TypeError):
        return (11, "other")

    if sic <= 9:
        return (8, "consumer_staples")      # agriculture, forestry, fishing
    elif sic <= 14:
        return (0, "energy")                # mining, oil & gas extraction
    elif sic <= 17:
        return (1, "materials_industrial")  # construction
    elif sic <= 21:
        return (8, "consumer_staples")      # food, tobacco
    elif sic <= 27:
        return (1, "materials_industrial")  # textiles, lumber, paper, printing
    elif sic == 28:
        return (4, "healthcare")            # chemicals, pharma, biotech
    elif sic == 29:
        return (0, "energy")                # petroleum refining
    elif sic <= 34:
        return (1, "materials_industrial")  # rubber, stone, metals, fabricated metals
    elif sic <= 36:
        return (2, "tech_hardware")         # computers, electronics, semiconductors
    elif sic <= 39:
        return (1, "materials_industrial")  # transport equip, instruments, misc mfg
    elif sic <= 47:
        return (1, "materials_industrial")  # transportation, logistics
    elif sic == 48:
        return (9, "telecom_media")         # communications
    elif sic == 49:
        return (10, "utilities")            # electric, gas, sanitary
    elif sic <= 59:
        return (7, "consumer_disc")         # wholesale + retail trade
    elif sic <= 64:
        return (5, "financials")            # banks, credit, insurance
    elif sic == 65:
        return (6, "real_estate")           # real estate
    elif sic <= 67:
        return (5, "financials")            # holding companies, investment services
    elif sic == 73:
        return (3, "software_services")     # business services (incl. software 7372)
    elif sic == 80:
        return (4, "healthcare")            # health services
    elif sic <= 89:
        return (7, "consumer_disc")         # services, entertainment, education
    else:
        return (11, "other")


# Manual overrides for ETFs in the regime tickers list.
# Kept in sync with REGIME_TICKERS in config.py — any ETF not listed here
# will use the Polygon SIC code if available, falling back to "other".
ETF_SECTOR_OVERRIDES: Dict[str, Tuple[int, str]] = {
    # Broad market
    "SPY": (11, "other"), "QQQ": (11, "other"),
    "IWM": (11, "other"), "MDY": (11, "other"),
    # Sector ETFs → map to their sector
    "XLK": (2, "tech_hardware"),
    "XLF": (5, "financials"),
    "XLE": (0, "energy"),
    "XLV": (4, "healthcare"),
    "XLI": (1, "materials_industrial"),
    "XLY": (7, "consumer_disc"),
    "XLP": (8, "consumer_staples"),
    "XLU": (10, "utilities"),
    "XLB": (1, "materials_industrial"),
    "XLRE": (6, "real_estate"),
    "XLC": (9, "telecom_media"),
    # Fixed income
    "TLT": (11, "other"), "IEF": (11, "other"), "SHY": (11, "other"),
    "LQD": (11, "other"), "HYG": (11, "other"),
    # Volatility
    "VIXY": (11, "other"), "VIXM": (11, "other"),
    # Dollar
    "UUP": (11, "other"),
    # Commodities
    "GLD": (11, "other"), "USO": (11, "other"),
    # International
    "EFA": (11, "other"), "EEM": (11, "other"), "EWJ": (11, "other"),
}
