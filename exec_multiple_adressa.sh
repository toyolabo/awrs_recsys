#!/bin/bash

# --- 1st run
python newsreclib/train.py experiment=nrms_adressaoneweek_plm_celoss_bertsent || echo "train failed"
python newsreclib/train.py experiment=naml_adressaoneweek_plm_celoss_bertsent || echo "train failed"
python newsreclib/train.py experiment=lsturini_adressaoneweek_plm_celoss_bertsent || echo "train failed"
python newsreclib/train.py experiment=tanr_adressaoneweek_plm_celoss_bertsent || echo "train failed"
python newsreclib/train.py experiment=sentirec_adressaoneweek_plm_celoss_bertsent || echo "train failed"


# --- 2nd run
python newsreclib/train.py experiment=nrms_adressaoneweek_plm_celoss_bertsent || echo "train failed"
python newsreclib/train.py experiment=naml_adressaoneweek_plm_celoss_bertsent || echo "train failed"
python newsreclib/train.py experiment=lsturini_adressaoneweek_plm_celoss_bertsent || echo "train failed"
python newsreclib/train.py experiment=tanr_adressaoneweek_plm_celoss_bertsent || echo "train failed"
python newsreclib/train.py experiment=sentirec_adressaoneweek_plm_celoss_bertsent || echo "train failed"

# --- 3rd run
python newsreclib/train.py experiment=nrms_adressaoneweek_plm_celoss_bertsent || echo "train failed"
python newsreclib/train.py experiment=naml_adressaoneweek_plm_celoss_bertsent || echo "train failed"
python newsreclib/train.py experiment=lsturini_adressaoneweek_plm_celoss_bertsent || echo "train failed"
python newsreclib/train.py experiment=tanr_adressaoneweek_plm_celoss_bertsent || echo "train failed"
python newsreclib/train.py experiment=sentirec_adressaoneweek_plm_celoss_bertsent || echo "train failed"