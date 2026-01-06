import requests
import json
import os
import sys

# Configuration
API_URL = "https://guido-2--dolphin-pdf-parser-web-app-dev.modal.run/api/convert"
SAMPLE_PDF = "demo/page_imgs/page_6.pdf"

def test_upload():
    if not os.path.exists(SAMPLE_PDF):
        print(f"❌ Error: Sample file '{SAMPLE_PDF}' not found.")
        return

    print(f"🚀 Uploading '{SAMPLE_PDF}' to {API_URL}...")
    
    try:
        with open(SAMPLE_PDF, "rb") as f:
            files = {"file": f}
            response = requests.post(API_URL, files=files)
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                print("\n✅ Success! Document processed.")
                print(f"   Total Pages: {result.get('total_pages')}")
                print("\n--- Extracted Markdown Start ---")
                print(result.get("markdown")[:500] + "...")
                print("--- Extracted Markdown End ---\n")
                
                # Save full result
                with open("test_result.md", "w") as f:
                    f.write(result.get("markdown", ""))
                print("📝 Full result saved to 'test_result.md'")
            else:
                print(f"\n❌ Processing failed: {result.get('error')}")
        else:
            print(f"\n❌ API Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    test_upload()
