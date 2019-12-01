from comet_ml import Experiment
import lightgbm as lgb

from pirates.visualization import visualize

hyperparameters = {
                'is_unbalance': True,
                }

def train_lgbm(x_train, x_val, y_train, y_val):
    # Make sure labels not one-hot

    experiment = Experiment(
            api_key="VNQSdbR1pw33EkuHbUsGUSZWr",
            project_name="piratesofthecaribbean",
            workspace="florpi",
            auto_param_logging=False,
        )
    experiment.log_parameters(hyperparameters)

    clf = lgb.LGBMClassifier(**hyperparameters)
    clf.fit(X=x_train, y=y_train, eval_set=(x_val, y_val),
                eval_metric='AUC')

    y_pred = clf.predict(x_val)
    visualize.plot_confusion_matrix(
        y_val,
        y_pred,
        classes=LABELS,
        normalize=True,
        experiment=experiment,
    )

    visualize.plot_confusion_matrix(
        y_val,
        y_pred,
        classes=LABELS,
        normalize=False,
        experiment=experiment,
    )
