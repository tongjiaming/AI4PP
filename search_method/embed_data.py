import torch
import time

from import_data import import_data
from unixcoder import UniXcoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UniXcoder("microsoft/unixcoder-base")
model.to(device)

data_path = "dataset/"
embedding_filename = "embeddings/embeddings"
data_filename = "embeddings/data"


def embed_data(data_path):
    time_start = time.time()

    data = import_data(data_path)

    print("embedding data...")

    i = 0

    for snippet in data:
        tokens_ids = model.tokenize([snippet], max_length=512, mode="<encoder-only>")
        source_ids = torch.tensor(tokens_ids).to(device)
        tokens_embeddings, data_embedding = model(source_ids)
        norm_data_embedding = torch.nn.functional.normalize(data_embedding, p=2, dim=1)
        torch.save(norm_data_embedding, embedding_filename + str(i) + '.pt')
        torch.save(snippet, data_filename + str(i) + '.pt')
        i = i + 1

    print("data embedding done in ", time.time() - time_start, "s!")
    print("embeddings located in ", embedding_filename)
    print("================================================================================================")


embed_data(data_path)
