import os, sys
from random import randint, seed
import torch
from generator_disco.generator import GeneratorDisco
from generator_ld.generator import GeneratorLatentDiffusion

class Chain:

    DEVICE = None
    device = None
    output_filename = None
    
    generator_disco = None
    generator_ld = None
    
    def load_cuda(self):
        
        self.DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Using device:', self.DEVICE)
        self.device = self.DEVICE # At least one of the modules expects this name..

        if torch.cuda.get_device_capability(self.DEVICE) == (8,0): ## A100 fix thanks to Emad
            print('Disabling CUDNN for A100 gpu', file=sys.stderr)
            torch.backends.cudnn.enabled = False
            
    def run_chain(self,prompt):
        
        print (prompt)
        run_ld = True
        run_disco = True
        
        if run_ld:
            self.generator_ld.args.prefix = str(randint(0,1000000))
            self.output_filename = self.generator_ld.do_run(prompt,self.generator_ld.args.prefix,str(100))

        if run_disco:
            self.generator_disco.settings["prompt"] = [prompt]
            if run_ld: 
                self.generator_disco.settings["skip_steps"] = 15
                self.generator_disco.settings["init_image"] = os.getcwd() + "/static/output/" + self.output_filename
            self.generator_disco.init_settings()
            self.output_filename = self.generator_disco.do_run()
        
        return self.output_filename

    def __init__(self):
        
        seed(1)
        self.load_cuda()
        self.generator_disco = GeneratorDisco(self,50,[512,512])
        self.generator_ld =  GeneratorLatentDiffusion(self)
        
        
        