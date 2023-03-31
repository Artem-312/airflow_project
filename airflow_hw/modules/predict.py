import json
import os
import glob
import pandas as pd
import dill


from datetime import datetime


path = os.environ.get('PROJECT_PATH', '/Users/artemdyachenko/airflow_hw')


def pull():
    list_of_files = glob.glob('/Users/artemdyachenko/airflow_hw/data/models/*.pkl')
    global latest_file
    latest_file = max(list_of_files, key=os.path.getctime)


def predx():
    pickled_model = dill.load(open(latest_file, 'rb'))
    output = pd.DataFrame(columns=['car_id', 'pred'])
    for filename in glob.glob('/Users/artemdyachenko/airflow_hw/data/test/*.json'):
        with open(filename) as mod:
            form = json.load(mod)
            df = pd.DataFrame.from_dict([form])
            df_pred = pickled_model.predict(df)
            result = {'car_id': df.iloc[0]['id'], 'pred': df_pred[0]}
            output = output.append(result, ignore_index=True)
    output.to_csv(f'/Users/artemdyachenko/airflow_hw/data/predictions/pred_{datetime.now().strftime("%Y%m%d%H%M")}.csv', index=False)


def predict():
    pull()
    predx()


if __name__ == '__main__':
    predict()
