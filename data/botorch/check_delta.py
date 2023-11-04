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
    ["Penicillin", 7],
    ["VehicleSafety", 5],
    ["VehicleSafety2K", 5],
    ["CarSideImpact", 7],
    ["WeldedBeam", 4],
    ["DiscBrake", 4],
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
