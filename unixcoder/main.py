import torch
import json
from fuzzywuzzy import fuzz
from unixcoder import UniXcoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UniXcoder("microsoft/unixcoder-base")
model.to(device)

correct = 0
total = 0

with open('test.json') as f:
    for line in f:
        data = json.loads(line)
        context = data["input"]
        tokens_ids = model.tokenize([context], max_length=512, mode="<encoder-decoder>")
        source_ids = torch.tensor(tokens_ids).to(device)
        prediction_ids = model.generate(source_ids, decoder_only=False, beam_size=3, max_length=128)
        predictions = model.decode(prediction_ids)
        predictions = [x.replace("<mask0>","").strip() for x in predictions[0]]

        # print("==========================================================")
        # print(data["gt"])
        # print([x.replace("<mask0>", "").strip() for x in predictions[0]])

        edit_sim = 0
        total = total + 1
        if data['gt'] in predictions:
            correct = correct + 1
        edit_sims = [fuzz.ratio(pred, data['gt']) for pred in predictions]
        edit_sim += max(edit_sims)

    print("pass@3: ", correct/total*100, "%")
    print("Edit sim: ", edit_sim/total*100)
