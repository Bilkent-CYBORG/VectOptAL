import numpy as np
from sklearn.preprocessing import StandardScaler


def get_delta_90(mu):
    n = mu.shape[0]
    Delta = np.zeros(n)
    for i in range(n):
        vi = mu[i, :].reshape(1, -1)
        difs = mu - vi
        difs[difs < 0] = 0
        smallmij = np.min(difs, axis=1)
        Delta[i] = np.max(smallmij)

    return Delta.reshape(-1,1)

problems = [
    ["SnAr", 4],     # nucleophilic aromatic substitution
    ["SnArHALF", 4],     # nucleophilic aromatic substitution
    ["VdV", 2],      # Van de Vusse reaction
    ["VdV2K", 2],      # Van de Vusse reaction
    ["PK1", 2],      # Paal-Knorr reaction, fixed temperature
    ["PK12K", 2],      # Paal-Knorr reaction, fixed temperature
    ["PK2", 3],      # Paal-Knorr reaction
    ["PK2K", 3],      # Paal-Knorr reaction
    ["Lactose", 2],  # isomerisation of lactose to lactulose
]
for problem_name, inp_dim in problems:
    data = np.load(f"{problem_name}.npy", allow_pickle=True)
    print("Name - Size:", f"{problem_name} - {len(data)}")

    out_dim = len(data[0]) - inp_dim

    output_scaler = StandardScaler()
    # Y_scaled = data[:, inp_dim:]
    Y_scaled = output_scaler.fit_transform(data[:, inp_dim:])

    delta = get_delta_90(Y_scaled)
    print(sorted(delta[np.nonzero(delta)])[:5])
    print("Avg. gap:", np.mean(delta[np.nonzero(delta)]))
    print()
