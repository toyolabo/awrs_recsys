#!/bin/bash

# --- 1st run
python newsreclib/train.py experiment=naml_mindsmall_pretrainedemb_celoss_bertsent || echo "train failed"
python newsreclib/train.py experiment=lsturini_mindsmall_pretrainedemb_celoss_bertsent || echo "train failed"
python newsreclib/train.py experiment=sentirec_mindsmall_pretrainedemb_celoss_bertsent || echo "train failed"
python newsreclib/train.py experiment=tanr_mindsmall_pretrainedemb_celoss_bertsent || echo "train failed"
python newsreclib/train.py experiment=caum_mindsmall_pretrainedemb_celoss_bertsent || echo "train failed"

# --- 2nd run
python newsreclib/train.py experiment=nrms_mindsmall_pretrainedemb_celoss_bertsent || echo "train failed"
python newsreclib/train.py experiment=naml_mindsmall_pretrainedemb_celoss_bertsent || echo "train failed"
python newsreclib/train.py experiment=lsturini_mindsmall_pretrainedemb_celoss_bertsent || echo "train failed"
python newsreclib/train.py experiment=sentirec_mindsmall_pretrainedemb_celoss_bertsent || echo "train failed"
python newsreclib/train.py experiment=tanr_mindsmall_pretrainedemb_celoss_bertsent || echo "train failed"
python newsreclib/train.py experiment=caum_mindsmall_pretrainedemb_celoss_bertsent || echo "train failed"