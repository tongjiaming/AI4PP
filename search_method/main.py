import torch

from import_data import import_data
from unixcoder import UniXcoder

# unixcoder config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UniXcoder("microsoft/unixcoder-base")
model.to(device)

# user configuration and input
data_path = "dataset/"
embedding_filename = "embeddings/embeddings"
data_filename = "embeddings/data"
query = "This framework generates nothing for each input sentence"
# data = import_data(data_path)
number_of_data = 11967
num_of_suggestsion = 10

# Encode query
tokens_ids = model.tokenize([query], max_length=512, mode="<encoder-only>")
source_ids = torch.tensor(tokens_ids).to(device)
tokens_embeddings, query_embedding = model(source_ids)
norm_query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)

# Encode data and calculate similarities
similarities = []
data = []
for i in range(number_of_data):
    norm_data_embedding = torch.load(embedding_filename + str(i) + '.pt')
    similarity = torch.einsum("ac,bc->ab", norm_query_embedding, norm_data_embedding).item()
    similarities.append(similarity)
    snippet = torch.load(data_filename + str(i) + '.pt')
    data.append(snippet)

# Output
sort = sorted(range(len(similarities)), key=lambda k: similarities[k], reverse=True)
for i in range(num_of_suggestsion):
    print("rank:", i + 1)
    print("suggestion:\n", data[sort[i]])
    print("similarity", similarities[sort[i]])
    print('------------------------------------------------------------------------------------------------')
