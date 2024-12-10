import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "./third_party/DINO_UI"))
import argparse
import torch
from third_party.DINO_UI.models.dino import build_dino 
from utils.arg_utils import create_dino_args

def main():
    args = create_dino_args()
    model, criterion, postprocessors = build_dino(args)    # DINO_4scale

# 使用示例:
if __name__ == '__main__':
    main()
 