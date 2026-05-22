#!/usr/bin/env python3
"""Test adaptive conversation flow with different legal topics"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_scenario(name, query, answers):
    print(f"\n{'='*60}")
    print(f"SCENARIO: {name}")
    print(f"{'='*60}")
    print(f"User: {query}")
    
    # Start
    r = requests.post(f"{BASE_URL}/consult/start", json={"query": query, "max_turns": 3})
    data = r.json()
    sid = data['session_id']
    print(f"Bot:  {data.get('question', 'N/A')}")
    
    for i, answer in enumerate(answers):
        print(f"User: {answer}")
        r = requests.post(f"{BASE_URL}/consult/answer", json={"session_id": sid, "answer": answer})
        data = r.json()
        if data.get('next_action') == 'ask':
            print(f"Bot:  {data.get('question', 'N/A')}")
        else:
            print(f"[FINAL REPORT GENERATED - {len(data.get('report',''))} chars]")
            # Print first 200 chars of the report
            rpt = data.get('report','')
            clean = rpt.encode('ascii', errors='replace').decode('ascii')
            for line in clean.split('\n')[:15]:
                print(f"  {line}")
            print("  ...")
            break

if __name__ == "__main__":
    print("Testing Adaptive Question Generation")
    print("="*60)
    
    # Test 1: Cyber/privacy - the user's exact scenario
    test_scenario(
        "Privacy / Chat Leak (CYBER)",
        "What happens legally if someone leaks private chats online?",
        ["WhatsApp chats were leaked on Instagram", "Yes, my ex-boyfriend did it", "Two days ago"]
    )
    
    # Test 2: Lost phone scenario
    test_scenario(
        "Lost Phone Kept by Finder",
        "If someone finds a lost phone and keeps it without trying to return it, is that considered theft?",
        ["It was an iPhone 15 lost at a coffee shop", "Yesterday morning", "I have the purchase receipt and IMEI"]
    )
    
    # Test 3: Consumer complaint
    test_scenario(
        "Consumer Complaint",
        "I bought a laptop online but received a defective product and the seller refuses to refund",
        ["Amazon, Rs. 65000", "Two weeks ago", "Yes I have the invoice and complaint emails"]
    )
    
    print(f"\n{'='*60}")
    print("All scenarios completed!")
    print("="*60)
