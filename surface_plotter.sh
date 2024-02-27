#!/bin/bash

# Define variables
CKPT_PATH="CKPTS/VIT-LORA-3CKPTS/" # Path to the folder where npz file is available
WHATTOPLOT="te_err" # What to plot: ['tr_loss', 'tr_err', 'te_loss', 'te_err', 'all']
SAVEAS="both" # Save the plot as: ['png', 'svg', 'both']
CONNECTOPTIMAS=true # Connect the optimas with a line or not
CONTOUR=true # Plot contour or not

# Ensure CUDA_VISIBLE_DEVICES is set for the script, not the conda activate command
CUDA_VISIBLE_DEVICES=0 python3 surface_plotter.py --path="$CKPT_PATH" \
                           --whatToPlot="$WHATTOPLOT" \
                           --saveAs="$SAVEAS" \
                           --connectOptimas="$CONNECTOPTIMAS" \
                            --contour="$CONTOUR"

# Comment out lines connectOptimas/contour if not needed