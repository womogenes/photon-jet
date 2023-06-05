def run_trial(data, Phi_sizes, F_sizes):
    n_features = 7
    n_particles = 960

    model = PFN(
        n_features=n_features,
        n_particles=n_particles,
        n_outputs=3,
        Phi_sizes=(128,) * 4 + (64,) * 3,
        F_sizes=(128,) * 4 + (64,) * 3
    )
