1. Before running this code, in hand we should have 3 checkpoints.
2. Then we will find loss surface -> run surface_finder.sh
    1. Pass DIR means folder location where we want to save our plotted surface file "plane.npz"
    2. Dataset, Datapath where we need to save.
    3. Model can be VIT-LORA MODEL is kept as LORA_BNN file, which generates and returns already required base model. (Based on what parameters is fine-tuned/frozen we need to update one line of code in Bayes_wrap.py file. `self.delta_models[i].freeze_module(exclude=["deltas"], set_state_dict=True)`, with an assumption apart from Lora remaining model is frozen.
    4. Then just update CKPT param in surface_finder.sh with 3 ckpts (Whose loss surface we want to find).
    5. Change CUDA_VISIBLE_DEVICES=0/1/.. based on what GPU user wants to run this program on. By default program runs for 40 points (Includes model's optima as well, so we dont need to modify inside code)
3. Then once plane.npz file is created -> run surface_plotter.sh
    1. Based on comments in Surface_plotter.sh adjust the required values and run it, to get loss surfaces in png/svg format.
