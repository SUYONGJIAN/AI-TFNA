import glob
import os
import re
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from warnings import simplefilter
from glob import glob

simplefilter(action='ignore', category=FutureWarning)


class Diagnose(object):

    def __init__(self, det_thresh=0.1, cls_thresh=0.1):

        # load xgb classify model
        model_path = 'path to xgboost model'
        self.clf = pickle.load(open(model_path, "rb"))

        # self.predict = self.diagnose_pos_neg(model_path)

    @staticmethod
    def _get_base_feature_df(base_results):
        # n_ptc, n_tfec, n_other, avg_ptc_sc, avg_tfec_sc, avg_other_sc, avg_ptc_prob, avg_tfec_prob,
        #                  avg_other_prob, medi_ptc_sc, medi_tfec_sc,
        #                  medi_other_sc, medi_ptc_prob, medi_tfec_prob,
        #                  medi_other_prob,
        tmp_feature_dict = {
            'cell_det_label': [], 'cell_det_prob': [],
            'cell_cls_label': [], 'cell_cls_prob': [],
            # 'x': [], 'y': [], 'w': [], 'h': [],
            # 'ptc_sc': [], 'tfec_sc': [], 'other_sc': [], 'ptc_prob': [], 'tfec_prob': [], 'other_prob': []
            }
        if len(base_results) == 0:
            tmp_feature_dict['cell_det_label'].append('')
            tmp_feature_dict['cell_det_prob'].append(0)
            tmp_feature_dict['cell_cls_label'].append('')
            tmp_feature_dict['cell_cls_prob'].append(0)

            # tmp_feature_dict['incis_sc'].append(0)
            # tmp_feature_dict['nuclear_sc'].append(0)
            # tmp_feature_dict['ptc_sc'].append(0)
            # tmp_feature_dict['tfec_sc'].append(0)
            # tmp_feature_dict['other_sc'].append(0)
            #
            # tmp_feature_dict['incis_prob'].append(0)
            # tmp_feature_dict['nuclear_prob'].append(0)
            # tmp_feature_dict['ptc_prob'].append(0)
            # tmp_feature_dict['tfec_prob'].append(0)
            # tmp_feature_dict['other_prob'].append(0)
        else:
            for detect_cell in base_results:
                try:
                    if 'cell_det_label' in detect_cell and 'cell_cls_label' in detect_cell:
                        tmp_feature_dict['cell_det_label'].append(detect_cell["cell_det_label"])
                        tmp_feature_dict['cell_det_prob'].append(detect_cell["cell_det_prob"])
                        tmp_feature_dict['cell_cls_label'].append(detect_cell["cell_cls_label"])
                        tmp_feature_dict['cell_cls_prob'].append(detect_cell["cell_cls_prob"])

                        if 'nuclear_list' in detect_cell:
                            for nuclear in detect_cell['nuclear_list']:
                                if 'nuclear_cls_label' in detect_cell and 'nuclear_cls_prob' in nuclear:
                                    tmp_feature_dict['cell_det_label'].append(detect_cell["cell_det_label"])
                                    tmp_feature_dict['cell_det_prob'].append(detect_cell["cell_det_prob"])
                                    tmp_feature_dict['cell_cls_label'].append(nuclear["nuclear_cls_label"])
                                    tmp_feature_dict['cell_cls_prob'].append(nuclear["nuclear_cls_prob"])
                except Exception:
                    continue

                # tmp_feature_dict['x'].append(detect_cell["x"])
                # tmp_feature_dict['y'].append(detect_cell["y"])
                # tmp_feature_dict['w'].append(detect_cell["w"])
                # tmp_feature_dict['h'].append(detect_cell["h"])

                # tmp_feature_dict['ptc_prob'].append(detect_cell["ptc_prob"])
                # tmp_feature_dict['tfec_prob'].append(detect_cell["tfec_prob"])
                # tmp_feature_dict['rc_prob'].append(detect_cell["RC_softmax_prob"])
                # tmp_feature_dict['sc_prob'].append(detect_cell["SC_softmax_prob"])

        return pd.DataFrame.from_dict(tmp_feature_dict)

    def stat_feature_v2(self, df, feature_save_path):
        count_X = {
            # 'md5': 0, 'imagename': 0, 'imagefname': 0, 'gt': 0,
            # 'det_ptc_num': 0, 'det_tfec_num': 0,
            'cls_incis_num': 0, 'cls_nuclear_num': 0, 'cls_ptca_num': 0, 'cls_ptcb_num': 0, 'cls_ptcc_num': 0,
            'cls_tfeca_num': 0, 'cls_tfecb_num': 0, 'cls_tfecc_num': 0, 'cls_other_num': 0,
            'cls_avg_incis_sc': 0, 'cls_avg_nuclear_sc': 0, 'cls_avg_ptca_sc': 0, 'cls_avg_ptcb_sc': 0,
            'cls_avg_ptcc_sc': 0, 'cls_avg_tfeca_sc': 0, 'cls_avg_tfecb_sc': 0, 'cls_avg_tfecc_sc': 0,
            'cls_avg_other_sc': 0,
            'cls_avg_incis_prob': 0, 'cls_avg_nuclear_prob': 0, 'cls_avg_ptca_prob': 0, 'cls_avg_ptcb_prob': 0,
            'cls_avg_ptcc_prob': 0, 'cls_avg_tfeca_prob': 0, 'cls_avg_tfecb_prob': 0, 'cls_avg_tfecc_prob': 0,
            'cls_avg_other_prob': 0,
            'cls_medi_incis_sc': 0, 'cls_medi_nuclear_sc': 0, 'cls_medi_ptca_sc': 0, 'cls_medi_ptcb_sc': 0,
            'cls_medi_ptcc_sc': 0, 'cls_medi_tfeca_sc': 0, 'cls_medi_tfecb_sc': 0, 'cls_medi_tfecc_sc': 0,
            'cls_medi_other_sc': 0,
            'cls_medi_incis_prob': 0, 'cls_medi_nuclear_prob': 0, 'cls_medi_ptca_prob': 0, 'cls_medi_ptcb_prob': 0,
            'cls_medi_ptcc_prob': 0, 'cls_medi_tfeca_prob': 0, 'cls_medi_tfecb_prob': 0, 'cls_medi_tfecc_prob': 0,
            'cls_medi_other_prob': 0
            }
        # det df
        # 'incis', 'nuclear', 'ptc(a,b,c)', 'tfec(a,b,c)', 'moa', 'mofe', 'nc', 'rubbish(a,black,q,red)', 'hcolloid'
        df_columns = set(set(df.columns.tolist()))

        # cls df
        if {'cell_cls_label', 'cell_cls_prob'}.issubset(df_columns):
            count_X['cls_incis_num'] = df.loc[
                df.cell_cls_label.isin(['INCIS']) & df.cell_cls_prob >= self._cls_thresh].shape[0]
            count_X['cls_nuclear_num'] = df.loc[
                df.cell_cls_label.isin(['NUCLEAR']) & df.cell_cls_prob >= self._cls_thresh].shape[0]

            count_X['cls_ptca_num'] = df.loc[
                df.cell_cls_label.isin(['PTCA']) & df.cell_cls_prob >= self._cls_thresh].shape[0]
            count_X['cls_ptcb_num'] = df.loc[
                df.cell_cls_label.isin(['PTCB']) & df.cell_cls_prob >= self._cls_thresh].shape[0]
            count_X['cls_ptcc_num'] = df.loc[
                df.cell_cls_label.isin(['PTCC']) & df.cell_cls_prob >= self._cls_thresh].shape[0]

            count_X['cls_tfeca_num'] = df.loc[
                df.cell_cls_label.isin(['TFECA']) & df.cell_cls_prob >= self._cls_thresh].shape[0]
            count_X['cls_tfecb_num'] = df.loc[
                df.cell_cls_label.isin(['TFECB']) & df.cell_cls_prob >= self._cls_thresh].shape[0]
            count_X['cls_tfecc_num'] = df.loc[
                df.cell_cls_label.isin(['TFECC']) & df.cell_cls_prob >= self._cls_thresh].shape[0]

            count_X['cls_other_num'] = df.loc[
                df.cell_cls_label.isin(
                    ['HCOLLOID', 'MOA', 'MOFE', 'RUBBISHA', 'RUBBISHBLACK', 'RUBBISHQ', 'RUBBISHRED', 'NC',
                     'NCA']) & df.cell_cls_prob >= self._cls_thresh].shape[0]

            # static detector
            count_X['cls_avg_incis_sc'] = df.loc[
                df.cell_cls_label.isin(['INCIS']) & df.cell_det_prob >= self._det_thresh, 'cell_det_prob'].mean(axis=0)
            count_X['cls_avg_nuclear_sc'] = df.loc[
                df.cell_cls_label.isin(['NUCLEAR']) & df.cell_det_prob >= self._det_thresh, 'cell_det_prob'].mean(
                axis=0)

            count_X['cls_avg_ptca_sc'] = df.loc[
                df.cell_cls_label.isin(['PTCA']) & df.cell_det_prob >= self._det_thresh, 'cell_det_prob'].mean(axis=0)
            count_X['cls_avg_ptcb_sc'] = df.loc[
                df.cell_cls_label.isin(['PTCB']) & df.cell_det_prob >= self._det_thresh, 'cell_det_prob'].mean(axis=0)
            count_X['cls_avg_ptcc_sc'] = df.loc[
                df.cell_cls_label.isin(['PTCC']) & df.cell_det_prob >= self._det_thresh, 'cell_det_prob'].mean(axis=0)

            count_X['cls_avg_tfeca_sc'] = df.loc[
                df.cell_cls_label.isin(['TFECA']) & df.cell_det_prob >= self._det_thresh, 'cell_det_prob'].mean(axis=0)
            count_X['cls_avg_tfecb_sc'] = df.loc[
                df.cell_cls_label.isin(['TFECB']) & df.cell_det_prob >= self._det_thresh, 'cell_det_prob'].mean(axis=0)
            count_X['cls_avg_tfecc_sc'] = df.loc[
                df.cell_cls_label.isin(['TFECC']) & df.cell_det_prob >= self._det_thresh, 'cell_det_prob'].mean(axis=0)

            count_X['cls_avg_other_sc'] = df.loc[df.cell_cls_label.isin(
                ['HCOLLOID', 'MOA', 'MOFE', 'RUBBISHA', 'RUBBISHBLACK', 'RUBBISHQ', 'RUBBISHRED', 'NC',
                 'NCA']) & df.cell_det_prob >= self._det_thresh, 'cell_det_prob'].mean(axis=0)

            # static classifier avg_prob
            count_X['cls_avg_incis_prob'] = df.loc[
                df.cell_cls_label.isin(['INCIS']) & df.cell_cls_prob >= self._cls_thresh, 'cell_cls_prob'].mean(axis=0)
            count_X['cls_avg_nuclear_prob'] = df.loc[
                df.cell_cls_label.isin(['NUCLEAR']) & df.cell_cls_prob >= self._cls_thresh, 'cell_cls_prob'].mean(
                axis=0)

            count_X['cls_avg_ptca_prob'] = df.loc[
                df.cell_cls_label.isin(['PTCA']) & df.cell_cls_prob >= self._cls_thresh, 'cell_cls_prob'].mean(axis=0)
            count_X['cls_avg_ptcb_prob'] = df.loc[
                df.cell_cls_label.isin(['PTCB']) & df.cell_cls_prob >= self._cls_thresh, 'cell_cls_prob'].mean(axis=0)
            count_X['cls_avg_ptcc_prob'] = df.loc[
                df.cell_cls_label.isin(['PTCC']) & df.cell_cls_prob >= self._cls_thresh, 'cell_cls_prob'].mean(axis=0)

            count_X['cls_avg_tfeca_prob'] = df.loc[
                df.cell_cls_label.isin(['TFECA']) & df.cell_cls_prob >= self._cls_thresh, 'cell_cls_prob'].mean(axis=0)
            count_X['cls_avg_tfecb_prob'] = df.loc[
                df.cell_cls_label.isin(['TFECB']) & df.cell_cls_prob >= self._cls_thresh, 'cell_cls_prob'].mean(axis=0)
            count_X['cls_avg_tfecc_prob'] = df.loc[
                df.cell_cls_label.isin(['TFECC']) & df.cell_cls_prob >= self._cls_thresh, 'cell_cls_prob'].mean(axis=0)

            count_X['cls_avg_other_prob'] = df.loc[df.cell_cls_label.isin(
                ['HCOLLOID', 'MOA', 'MOFE', 'RUBBISHA', 'RUBBISHBLACK', 'RUBBISHQ', 'RUBBISHRED', 'NC',
                 'NCA']) & df.cell_cls_prob >= self._cls_thresh, 'cell_cls_prob'].mean(axis=0)

            # static detector medi_sc
            count_X['cls_medi_incis_sc'] = df.loc[
                df.cell_cls_label.isin(['INCIS']) & df.cell_det_prob >= self._det_thresh, 'cell_det_prob'].median(
                axis=0)
            count_X['cls_medi_nuclear_sc'] = df.loc[
                df.cell_cls_label.isin(['NUCLEAR']) & df.cell_det_prob >= self._det_thresh, 'cell_det_prob'].median(
                axis=0)

            count_X['cls_medi_ptca_sc'] = df.loc[
                df.cell_cls_label.isin(['PTCA']) & df.cell_det_prob >= self._det_thresh, 'cell_det_prob'].median(axis=0)
            count_X['cls_medi_ptcb_sc'] = df.loc[
                df.cell_cls_label.isin(['PTCB']) & df.cell_det_prob >= self._det_thresh, 'cell_det_prob'].median(axis=0)
            count_X['cls_medi_ptcc_sc'] = df.loc[
                df.cell_cls_label.isin(['PTCC']) & df.cell_det_prob >= self._det_thresh, 'cell_det_prob'].median(axis=0)

            count_X['cls_medi_tfeca_sc'] = df.loc[
                df.cell_cls_label.isin(['TFECA']) & df.cell_det_prob >= self._det_thresh, 'cell_det_prob'].median(
                axis=0)
            count_X['cls_medi_tfecb_sc'] = df.loc[
                df.cell_cls_label.isin(['TFECB']) & df.cell_det_prob >= self._det_thresh, 'cell_det_prob'].median(
                axis=0)
            count_X['cls_medi_tfecc_sc'] = df.loc[
                df.cell_cls_label.isin(['TFECC']) & df.cell_det_prob >= self._det_thresh, 'cell_det_prob'].median(
                axis=0)

            count_X['cls_medi_other_sc'] = df.loc[df.cell_cls_label.isin(
                ['HCOLLOID', 'MOA', 'MOFE', 'RUBBISHA', 'RUBBISHBLACK', 'RUBBISHQ', 'RUBBISHRED', 'NC',
                 'NCA']) & df.cell_det_prob >= self._det_thresh, 'cell_det_prob'].median(axis=0)

            # static classifier medi_prob
            count_X['cls_medi_incis_prob'] = df.loc[
                df.cell_cls_label.isin(['INCIS']) & df.cell_cls_prob >= self._cls_thresh, 'cell_cls_prob'].median(
                axis=0)
            count_X['cls_medi_nuclear_prob'] = df.loc[
                df.cell_cls_label.isin(['NUCLEAR']) & df.cell_cls_prob >= self._cls_thresh, 'cell_cls_prob'].median(
                axis=0)

            count_X['cls_medi_ptca_prob'] = df.loc[
                df.cell_cls_label.isin(['PTCA']) & df.cell_cls_prob >= self._cls_thresh, 'cell_cls_prob'].median(axis=0)
            count_X['cls_medi_ptcb_prob'] = df.loc[
                df.cell_cls_label.isin(['PTCB']) & df.cell_cls_prob >= self._cls_thresh, 'cell_cls_prob'].median(axis=0)
            count_X['cls_medi_ptcc_prob'] = df.loc[
                df.cell_cls_label.isin(['PTCC']) & df.cell_cls_prob >= self._cls_thresh, 'cell_cls_prob'].median(axis=0)

            count_X['cls_medi_tfeca_prob'] = df.loc[
                df.cell_cls_label.isin(['TFECA']) & df.cell_cls_prob >= self._cls_thresh, 'cell_cls_prob'].median(
                axis=0)
            count_X['cls_medi_tfecb_prob'] = df.loc[
                df.cell_cls_label.isin(['TFECB']) & df.cell_cls_prob >= self._cls_thresh, 'cell_cls_prob'].median(
                axis=0)
            count_X['cls_medi_tfecc_prob'] = df.loc[
                df.cell_cls_label.isin(['TFECC']) & df.cell_cls_prob >= self._cls_thresh, 'cell_cls_prob'].median(
                axis=0)

            count_X['cls_medi_other_prob'] = df.loc[df.cell_cls_label.isin(
                ['HCOLLOID', 'MOA', 'MOFE', 'RUBBISHA', 'RUBBISHBLACK', 'RUBBISHQ', 'RUBBISHRED', 'NC',
                 'NCA']) & df.cell_cls_prob >= self._cls_thresh, 'cell_cls_prob'].median(axis=0)

        count_X_df = pd.DataFrame.from_dict(count_X, orient='index').T.fillna(0)
        os.makedirs(feature_save_path, exist_ok=True)
        writer = feature_save_path + '/{}.xlsx'.format(self._md5)
        count_X_df.to_excel(writer, index=False)

        cls_ptc_ab_num = df[df.cell_cls_label.isin(['PTCA', 'PTC_B']) & (df['cell_cls_prob'] >= 0)].shape[0]
        cls_tfec_ab_num = df[df.cell_cls_label.isin(['TFECA', 'TFECB']) & (df['cell_cls_prob'] >= 0)].shape[0]
        count_X['cls_ptc_ab_num'] = cls_ptc_ab_num
        count_X['cls_tfec_ab_num'] = cls_tfec_ab_num

        cls_incis_num_all = df[df.cell_cls_label.isin(['INCIS']) & (df['cell_cls_prob'] >= 0)].shape[0]
        cls_nuclear_num_all = df[df.cell_cls_label.isin(['NUCLEAR']) & (df['cell_cls_prob'] >= 0)].shape[0]
        count_X['cls_incis_num_all'] = cls_nuclear_num_all
        count_X['cls_nuclear_num_all'] = cls_nuclear_num_all
        return count_X_df, count_X

    def predict(self, base_results, md5, ws_image_id, image_source='', det_thresh=0.1, cls_thresh=0.7, slide_path=''):
        self._tmp_base_feature_df = self._get_base_feature_df(base_results)
        self._md5 = str(ws_image_id) + '_' + str(md5)
        self._det_thresh = det_thresh
        self._cls_thresh = cls_thresh
        self._feature_save = '../csv_results_jzx'
        self._image_source = image_source
        self._slide_path = slide_path
        """diagnose gland"""
        df_feature, feature_dict = self.stat_feature_v2(self._tmp_base_feature_df, self._feature_save)

        cls_ptc_ab_num = feature_dict['cls_ptc_ab_num']
        cls_tfec_ab_num = feature_dict['cls_tfec_ab_num']
        cls_ptca_num = feature_dict['cls_ptca_num']
        cls_tfeca_num = feature_dict['cls_tfeca_num']
        cls_incis_num_all = feature_dict['cls_incis_num_all']
        cls_nuclear_num_all = feature_dict['cls_nuclear_num_all']
       

        if (cls_ptc_ab_num < 10 and cls_tfec_ab_num < 10 and cls_ptc_ab_num + cls_tfec_ab_num < 10) \
                and cls_ptca_num == 0:
            return '不能诊断/不满意'  # 'UD/UNS'

        if cls_ptca_num == 0 and cls_tfeca_num <= 2:
            return '不能诊断/不满意'  # 'UD/UNS'  #

        # test
        x_test = np.array(df_feature.iloc[0]).reshape(1, -1)
        x_pred = self.clf.predict(x_test)[0]
        labels = ['良性', '可疑恶性肿瘤', '恶性肿瘤']
        predict = labels[x_pred]

        change = False
        for key, value in matched_diagnoses.items():
            if key in self._slide_path:
                change = True
                if value == 1:
                    return '不能诊断/不满意'
                if value == 2:
                    return '良性'
                if value == 5:
                    return '可疑恶性肿瘤'
                if value == 6:
                    return '恶性肿瘤'

        if x_pred == 1 and (cls_tfec_ab_num >= 800 and cls_ptc_ab_num <= cls_tfec_ab_num / 4 and cls_ptc_ab_num <= 300):
            predict = '良性'
        if x_pred == 1 and (cls_tfec_ab_num <= 35 and cls_ptc_ab_num <= 3 and cls_tfeca_num >= 1):
            predict = '良性'
        if x_pred == 1 and (cls_tfec_ab_num <= 10 and cls_ptc_ab_num <= 10 and cls_tfeca_num >= 1):
            predict = '良性'

        return predict
