import optuna


def objective(trial):
    x = trial.suggest_uniform("x", -100, 100)
    y = trial.suggest_categorical("y", [-1, 0, 1])
    return x ** 2 + y


study = optuna.create_study()
study.optimize(objective, n_trials=30)

optuna.visualization.plot_contour(study, params=["x", "y"])
