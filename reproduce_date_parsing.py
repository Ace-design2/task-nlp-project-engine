from inference import analyze
import json

def test_deadline_parsing():
    # Test cases including the problematic "this evening"
    text = "call my mom tomorrow and get my car fixed this evening and finish work tonight"
    print(f"Input: {text}")
    results = analyze(text)
    print("Output:")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    test_deadline_parsing()
