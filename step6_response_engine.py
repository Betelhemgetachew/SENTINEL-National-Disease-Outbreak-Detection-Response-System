# ============================================================
# SENTINEL — National Disease Outbreak Detection & Response System
# Step 6: Response Engine
# ============================================================
# For every alert fired in Step 4, this module:
#   1. Recommends a specific response action
#   2. Identifies who should be notified
#   3. Generates a notification payload (simulated)
#   4. Produces a response report
#
# HOW TO RUN:
#   python step6_response_engine.py
#
# MUST run steps 1–5 first
# ============================================================

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

DATA_DIR   = "clean_data"
OUTPUT_DIR = "outputs"
DASH_DIR   = "dashboard_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DASH_DIR,   exist_ok=True)

print("=" * 60)
print("STEP 6: RESPONSE ENGINE")
print("=" * 60)

# ── Load alerts ───────────────────────────────────────────────────────────────
alerts = pd.read_csv(f"{DATA_DIR}/alerts_2024.csv")
print(f"Loaded {len(alerts)} alerts to process")

# ── Response contacts per county ──────────────────────────────────────────────
# In a real system these would come from a contacts database
COUNTY_CONTACTS = {
    'Nairobi':    {'officer': 'Dr. Jane Mwangi',       'email': 'jmwangi@nairobi.health.go.ke',    'phone': '+254700000001'},
    'Mombasa':    {'officer': 'Dr. Ali Hassan',         'email': 'ahassan@mombasa.health.go.ke',    'phone': '+254700000002'},
    'Kisumu':     {'officer': 'Dr. Grace Odhiambo',     'email': 'godhiambo@kisumu.health.go.ke',   'phone': '+254700000003'},
    'Nakuru':     {'officer': 'Dr. Peter Kamau',        'email': 'pkamau@nakuru.health.go.ke',      'phone': '+254700000004'},
    'Garissa':    {'officer': 'Dr. Fatuma Omar',        'email': 'fomar@garissa.health.go.ke',      'phone': '+254700000005'},
    'Turkana':    {'officer': 'Dr. Simon Ekiru',        'email': 'sekiru@turkana.health.go.ke',     'phone': '+254700000006'},
    'Kitale':     {'officer': 'Dr. Mary Chebet',        'email': 'mchebet@kitale.health.go.ke',     'phone': '+254700000007'},
    'Malindi':    {'officer': 'Dr. Omar Shafii',        'email': 'oshafii@malindi.health.go.ke',    'phone': '+254700000008'},
    'Nyeri':      {'officer': 'Dr. Susan Wanjiku',      'email': 'swanjiku@nyeri.health.go.ke',     'phone': '+254700000009'},
    'Machakos':   {'officer': 'Dr. John Mutua',         'email': 'jmutua@machakos.health.go.ke',    'phone': '+254700000010'},
    'Meru':       {'officer': 'Dr. Agnes Karimi',       'email': 'akarimi@meru.health.go.ke',       'phone': '+254700000011'},
    'Thika':      {'officer': 'Dr. David Njoroge',      'email': 'dnjoroge@thika.health.go.ke',     'phone': '+254700000012'},
    'Kisii':      {'officer': 'Dr. Rose Nyamweya',      'email': 'rnyamweya@kisii.health.go.ke',    'phone': '+254700000013'},
    'Kakamega':   {'officer': 'Dr. Eric Wafula',        'email': 'ewafula@kakamega.health.go.ke',   'phone': '+254700000014'},
    'Eldoret':    {'officer': 'Dr. Lilian Chepkemoi',   'email': 'lchepkemoi@eldoret.health.go.ke', 'phone': '+254700000015'},
}

DEFAULT_CONTACT = {'officer': 'County Health Officer', 'email': 'health@county.go.ke', 'phone': '+254700000000'}

# ── National contacts (always notified for HIGH and CRITICAL) ─────────────────
NATIONAL_CONTACTS = [
    {'role': 'MoH Director of Public Health',       'email': 'dph@health.go.ke',         'phone': '+254200000001'},
    {'role': 'Kenya CDC Director',                  'email': 'director@kenicdc.go.ke',    'phone': '+254200000002'},
    {'role': 'WHO Kenya Country Representative',    'email': 'whokenya@who.int',          'phone': '+254200000003'},
]

# ── Response protocols per risk level ─────────────────────────────────────────
RESPONSE_PROTOCOLS = {
    'WATCH': {
        'title':   'Enhanced Surveillance',
        'actions': [
            'Increase reporting frequency from weekly to daily in affected county',
            'Alert county health team to monitor closely',
            'Review vaccination coverage in affected sub-counties',
            'Prepare rapid response team on standby',
        ],
        'timeframe':    'Within 48 hours',
        'notify_national': False,
    },
    'WARNING': {
        'title':   'Active Investigation',
        'actions': [
            'Deploy county rapid response team for field investigation',
            'Conduct active case finding in affected areas',
            'Collect samples for laboratory confirmation',
            'Identify source of outbreak (water, food, vectors)',
            'Issue health advisory to county hospitals and clinics',
        ],
        'timeframe':    'Within 24 hours',
        'notify_national': False,
    },
    'HIGH': {
        'title':   'Outbreak Containment',
        'actions': [
            'Activate county emergency operations center',
            'Deploy national rapid response team to support county',
            'Implement targeted vaccination or chemoprophylaxis if applicable',
            'Establish isolation/treatment centers if needed',
            'Issue public health advisory via media',
            'Coordinate with neighboring counties for border surveillance',
            'Submit situation report to Ministry of Health within 24 hours',
        ],
        'timeframe':    'Within 12 hours',
        'notify_national': True,
    },
    'CRITICAL': {
        'title':   'Emergency Response',
        'actions': [
            '🔴 ACTIVATE National Emergency Operations Center immediately',
            'Deploy national rapid response team within 6 hours',
            'Establish emergency treatment centers in affected areas',
            'Implement mass vaccination campaign if applicable',
            'Issue national public health emergency declaration',
            'Coordinate with WHO, CDC, and international partners',
            'Deploy military/civil resources for logistics support',
            'Implement movement controls if necessary',
            'Daily situation reports to Cabinet Secretary for Health',
            'Notify WHO under International Health Regulations (IHR 2005)',
        ],
        'timeframe':    'IMMEDIATE — Within 6 hours',
        'notify_national': True,
    }
}

# ── Disease-specific response guidance ────────────────────────────────────────
DISEASE_GUIDANCE = {
    'Malaria': {
        'key_action':   'Distribute insecticide-treated bed nets and indoor residual spraying',
        'lab_test':     'Rapid Diagnostic Test (RDT) or microscopy',
        'treatment':    'Artemisinin-based combination therapy (ACT)',
        'prevention':   'Vector control, bed nets, prophylaxis for high-risk groups',
    },
    'Typhoid': {
        'key_action':   'Investigate water and food sources; enforce water treatment',
        'lab_test':     'Blood culture, Widal test',
        'treatment':    'Azithromycin or fluoroquinolones',
        'prevention':   'Safe water, sanitation, hygiene (WASH), vaccination',
    },
    'Dengue': {
        'key_action':   'Vector control — eliminate Aedes mosquito breeding sites',
        'lab_test':     'NS1 antigen test, IgM/IgG serology',
        'treatment':    'Supportive care, fluid management',
        'prevention':   'Eliminate standing water, use repellents, protective clothing',
    },
    'Tuberculosis': {
        'key_action':   'Contact tracing and active case finding in community',
        'lab_test':     'Sputum smear microscopy, GeneXpert MTB/RIF',
        'treatment':    '6-month DOTS regimen (RHZE/RH)',
        'prevention':   'BCG vaccination, infection control, ventilation',
    },
    'Measles': {
        'key_action':   'Emergency mass vaccination campaign for children 6m–15 years',
        'lab_test':     'Serology (IgM), PCR',
        'treatment':    'Supportive care, Vitamin A supplementation',
        'prevention':   'MMR vaccination (target >95% coverage)',
    },
    'Meningitis': {
        'key_action':   'Prophylactic antibiotics for close contacts; mass vaccination',
        'lab_test':     'CSF analysis, blood culture, PCR',
        'treatment':    'IV penicillin or ceftriaxone',
        'prevention':   'Meningococcal vaccine, chemoprophylaxis for contacts',
    },
    'Diarrheal diseases': {
        'key_action':   'Emergency water treatment and sanitation improvement',
        'lab_test':     'Stool culture, cholera RDT',
        'treatment':    'Oral rehydration therapy (ORT), antibiotics for severe cases',
        'prevention':   'Safe water, handwashing, sanitation, oral cholera vaccine',
    },
}

# ── Generate response for each alert ─────────────────────────────────────────
print("\nGenerating response actions and notifications...")

response_records = []

for _, alert in alerts.iterrows():
    risk_level = alert['risk_level']
    county     = alert['county']
    disease    = alert['disease']
    protocol   = RESPONSE_PROTOCOLS[risk_level]
    guidance   = DISEASE_GUIDANCE.get(disease, {})
    contact    = COUNTY_CONTACTS.get(county, DEFAULT_CONTACT)

    # Build notification payload
    recipients = [
        {
            'role':    f"County Health Officer — {county}",
            'name':    contact['officer'],
            'email':   contact['email'],
            'phone':   contact['phone'],
            'channel': ['SMS', 'Email']
        }
    ]

    # National contacts for HIGH and CRITICAL
    if protocol['notify_national']:
        for nc in NATIONAL_CONTACTS:
            recipients.append({
                'role':    nc['role'],
                'name':    nc['role'],
                'email':   nc['email'],
                'phone':   nc['phone'],
                'channel': ['Email', 'SMS']
            })

    # Notification message
    message = (
        f"[SENTINEL {risk_level} ALERT] "
        f"{disease} outbreak detected in {county} County. "
        f"Cases are {alert['pct_above']:.0f}% above baseline "
        f"({int(alert['actual_cases'])} cases vs expected {int(alert['baseline'])}). "
        f"Required action: {protocol['title']} — {protocol['timeframe']}. "
        f"Login to SENTINEL dashboard for full details."
    )

    response_records.append({
        'alert_id':           f"ALT-2024-{len(response_records)+1:04d}",
        'timestamp':          alert['timestamp'],
        'county':             county,
        'disease':            disease,
        'risk_level':         risk_level,
        'actual_cases':       int(alert['actual_cases']),
        'baseline':           float(alert['baseline']),
        'pct_above':          float(alert['pct_above']),

        # Response protocol
        'response_title':     protocol['title'],
        'response_timeframe': protocol['timeframe'],
        'response_actions':   protocol['actions'],

        # Disease guidance
        'key_action':         guidance.get('key_action', 'Follow standard outbreak protocol'),
        'lab_test':           guidance.get('lab_test', 'Refer to national guidelines'),
        'treatment':          guidance.get('treatment', 'Refer to national guidelines'),
        'prevention':         guidance.get('prevention', 'Refer to national guidelines'),

        # Notifications
        'recipients':         recipients,
        'notification_msg':   message,
        'notification_sent':  True,   # simulated
        'notify_national':    protocol['notify_national'],
    })

response_df = pd.DataFrame(response_records)
print(f"✅ Generated {len(response_df)} response records")

# ── Print Response Report ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("🚨 SENTINEL RESPONSE REPORT — 2024")
print("=" * 60)

for level in ['CRITICAL', 'HIGH', 'WARNING', 'WATCH']:
    level_resp = response_df[response_df['risk_level'] == level]
    if len(level_resp) == 0:
        continue

    icon = {'CRITICAL':'🔴','HIGH':'🟠','WARNING':'🟡','WATCH':'🔵'}[level]
    print(f"\n{icon} {level} — {RESPONSE_PROTOCOLS[level]['title'].upper()}")
    print(f"   Timeframe: {RESPONSE_PROTOCOLS[level]['timeframe']}")
    print(f"   Total alerts: {len(level_resp)}")
    print(f"   Actions required:")
    for action in RESPONSE_PROTOCOLS[level]['actions'][:3]:
        print(f"     • {action}")

    print(f"\n   Sample alert:")
    sample = level_resp.iloc[0]
    print(f"     County:  {sample['county']}")
    print(f"     Disease: {sample['disease']}")
    print(f"     Cases:   {sample['actual_cases']:,} ({sample['pct_above']:.0f}% above baseline)")
    print(f"     Notified: {', '.join([r['role'].split('—')[0].strip() for r in sample['recipients'][:2]])}")
    print(f"     Message preview:")
    print(f"     \"{sample['notification_msg'][:120]}...\"")

# ── Summary statistics ────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("RESPONSE SUMMARY")
print("─" * 60)
total_notified = sum(len(r['recipients']) for r in response_records)
national_alerts = len(response_df[response_df['notify_national'] == True])
print(f"  Total responses generated:     {len(response_df)}")
print(f"  Total notifications sent:      {total_notified}")
print(f"  National-level escalations:    {national_alerts}")
print(f"  Counties with active response: {response_df['county'].nunique()}")
print(f"  Diseases under response:       {response_df['disease'].nunique()}")

# ── Save outputs ──────────────────────────────────────────────────────────────
# Save full response records as JSON for dashboard
with open(f"{DASH_DIR}/responses.json", 'w') as f:
    json.dump(response_records, f, indent=2)

# Save simplified CSV for analysis
response_df[[
    'alert_id','timestamp','county','disease','risk_level',
    'actual_cases','baseline','pct_above',
    'response_title','response_timeframe','key_action',
    'notification_sent','notify_national'
]].to_csv(f"{DATA_DIR}/responses_2024.csv", index=False)

print(f"\n✅ Response records saved: {DASH_DIR}/responses.json")
print(f"✅ Response CSV saved:     {DATA_DIR}/responses_2024.csv")

# ── Final Summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 COMPLETE ✅")
print("=" * 60)
print(f"""
SENTINEL now fully achieves all 3 requirements:

  ✅ DETECTION  — Prophet + Z-score anomaly detection
  ✅ MONITORING — 47 counties, 7 diseases, real-time tracking
  ✅ RESPONSE   — Tiered protocols, notifications, disease guidance

Run step5_export_dashboard_data.py again to include
response data in the dashboard JSON files.

ALL STEPS COMPLETE 🎉
""")
