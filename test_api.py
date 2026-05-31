#!/usr/bin/env python3
"""
Simple API key and connectivity test for SiliconFlow.
"""

import os
import sys

def test_api_connection():
    """Test if SiliconFlow API key is valid and we can reach the API."""
    
    api_key = os.environ.get("SILICONFLOW_API_KEY")
    if not api_key:
        print("❌ SILICONFLOW_API_KEY not set")
        print("\nSet it with:")
        print("  export SILICONFLOW_API_KEY='sk-your-actual-key'")
        return False
    
    print(f"✓ Found API key (length: {len(api_key)})")
    
    try:
        from openai import OpenAI
    except ImportError:
        print("❌ openai package not installed")
        print("Install with: pip3 install openai")
        return False
    
    print("✓ openai package imported")
    
    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.siliconflow.cn/v1"
        )
        print("✓ OpenAI client created")
        
        # Try a simple completion
        response = client.chat.completions.create(
            model="Qwen/Qwen3-8B",
            messages=[{"role": "user", "content": "Say 'Hello' only, nothing else."}],
            max_tokens=10,
            temperature=0.0,
        )
        
        reply = response.choices[0].message.content
        print(f"✓ API call successful! Response: {reply}")
        return True
        
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "invalid" in error_msg.lower():
            print(f"❌ API key error: {error_msg}")
            print("\nCheck your key:\n  1. Not expired\n  2. Copied correctly (no spaces/quotes)")
        else:
            print(f"❌ API error: {error_msg}")
        return False


if __name__ == "__main__":
    print("Testing SiliconFlow API connection...\n")
    success = test_api_connection()
    sys.exit(0 if success else 1)
