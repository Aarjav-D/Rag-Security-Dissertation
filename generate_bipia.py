import json
import os
import itertools
import pandas as pd

def generate_bipia_full(bipia_path="data/BIPIA/benchmark"):
    print("Generating full BIPIA dataset...")
    
    records = []
    
    # text tasks - email and table use text attacks
    text_tasks = {
        "email": "email",
        "table": "table"
    }
    
    for task, folder in text_tasks.items():
        for split in ["train", "test"]:
            context_file = os.path.join(bipia_path, folder, f"{split}.jsonl")
            attack_file = os.path.join(bipia_path, f"text_attack_{split}.json")
            
            if not os.path.exists(context_file) or not os.path.exists(attack_file):
                continue
            
            with open(context_file) as f:
                contexts = [json.loads(l) for l in f if l.strip()]
            
            with open(attack_file) as f:
                attacks = json.load(f)
            
            # flatten all attack instructions
            all_attacks = []
            for category, instructions in attacks.items():
                for instr in instructions:
                    all_attacks.append({
                        "instruction": instr,
                        "category": category
                    })
            
            # combine each context with each attack
            for ctx in contexts:
                base_text = ctx.get("context", ctx.get("passage", ""))
                if not base_text:
                    continue
                for attack in all_attacks:
                    # inject the attack at the end of the context
                    injected = base_text + "\n\n" + attack["instruction"]
                    records.append({
                        "text": injected[:1000],
                        "label": 1,
                        "source": "bipia",
                        "doc_type": task,
                        "attack_category": attack["category"]
                    })
    
    # code task uses code attacks
    for split in ["train", "test"]:
        context_file = os.path.join(bipia_path, "code", f"{split}.jsonl")
        attack_file = os.path.join(bipia_path, f"code_attack_{split}.json")
        
        if not os.path.exists(context_file) or not os.path.exists(attack_file):
            continue
        
        with open(context_file) as f:
            contexts = [json.loads(l) for l in f if l.strip()]
        
        with open(attack_file) as f:
            attacks = json.load(f)
        
        all_attacks = []
        for category, instructions in attacks.items():
            for instr in instructions:
                all_attacks.append({
                    "instruction": instr,
                    "category": category
                })
        
        for ctx in contexts:
            base_text = ctx.get("context", ctx.get("passage", ""))
            if isinstance(base_text, list):
                base_text = " ".join(str(x) for x in base_text)
            if not base_text:
                continue
            for attack in all_attacks:
                injected = base_text + "\n\n" + attack["instruction"]
                records.append({
                    "text": injected[:1000],
                    "label": 1,
                    "source": "bipia",
                    "doc_type": "code",
                    "attack_category": attack["category"]
                })
    
    print(f"  Generated {len(records)} BIPIA injection examples")
    return records

records = generate_bipia_full()
df = pd.DataFrame(records)
print(f"\nBreakdown by task:")
print(df["doc_type"].value_counts())
print(f"\nBreakdown by attack category:")
print(df["attack_category"].value_counts())