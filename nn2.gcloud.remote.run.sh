export REGION=us-east1
export BUCKET_NAME=gpu-quora
export JOB_NAME="deepnet2_$(date +%Y%m%d_%H%M%S)"
export OUTPUT_PATH=gs://$BUCKET_NAME/$JOB_NAME


gcloud ml-engine jobs submit training $JOB_NAME \
       --job-dir $OUTPUT_PATH \
       --package-path ./trainer \
       --module-name trainer.deepnet2 \
       --region $REGION \
       --runtime-version 1.0 \
       --config trainer/cloudml-gpu.yaml \
       -- \
       --gcp \
       --param_config gs://$BUCKET_NAME/nn2.conf \
       --data_path gs://$BUCKET_NAME/data/all_data.pickle
