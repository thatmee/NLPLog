import gc
import statistics
import datasets
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from typing import Union
from nlplog.config import Config
from nlplog.dataset import FailurePredDataset, SequencePredDataset


def votes_winner(subdf:pd.DataFrame):
    """
    One log has three predictions according to three reasons text and 
    the label that appears most often will be the final prediction of this log.

    Parameters:
        subdf: a dataframe that contains three lines of a log.
    """
    res = statistics.mode(subdf.predictions.to_list())
    return res


def calculate_score_by_log(
        predictions,
        val_data:datasets.Dataset,
        config:Config
    ):
    if config.task_type is None:
        if 'sentence_embeddings' not in val_data.column_names:
            config.task_type = 'other'
        elif val_data[0]['sentence_embeddings'].dim() == 2:
            config.task_type = 'failure_prediction'
        else:
            config.task_type = 'sequence_anomaly_detection'

    val_data = val_data.to_pandas()

    if len(val_data) != len(predictions):
        config.logger.warning(f'length of predictions({len(predictions)}) is not compatible with valid data({len(val_data)}). quit calculate.')

    val_data['predictions'] = predictions
    
    # voting is not good, delete this
    # res_data = val_data.groupby(['line_id', 'labels', 'template'], as_index=True).apply(votes_winner).reset_index(name='predictions')

    res_data = val_data

    if config.task_type == 'failure_prediction' and config.weight_metric:
        if 'template_cnt' not in val_data.columns:
            weight_df = pd.read_csv(r'/data/user/nyf/LAB/LogTurbo/data/bgl/bgl_template_weight.csv', usecols=['template', 'template_cnt'])
            res_data = val_data.merge(weight_df, how='left', on='template')
            # print(val_data)
            del weight_df
            gc.collect()
        # duplicate lines according to template_cnt
        res_data = res_data.loc[res_data.index.repeat(res_data['template_cnt'])].assign(template_cnt=1).reset_index(drop=True)        

    report = {}
    if config.confusion_matrix_enable:
        report['confusion_matrix'] = confusion_matrix(res_data['labels'], res_data['predictions'])
    for report_type in config.f1_report:
        precision, recall, f1, _ = precision_recall_fscore_support(res_data['labels'], res_data['predictions'], average=report_type, zero_division=0.0)
        report[report_type] = {'precision': precision, 'recall': recall, 'f1': f1}
    return report