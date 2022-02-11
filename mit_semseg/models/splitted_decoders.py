import torch
import torch.nn as nn
from mit_semseg.lib.nn import SynchronizedBatchNorm2d
BatchNorm2d = SynchronizedBatchNorm2d

# start, n_layers e final_conv sono i parametri aggiunti da me per operare la condivisione di certi layer del decoder
# start può essere True o False, e serve per indicare se il decoder che si sta creando è la parte iniziale o finale del decoder stesso
# n_layers indica quanti dei layer paralleli prendere in considerazione (se start == True allora vengono presi i primi n_layers,
# se invece start == False allora vengono presi gli ultimi n_layers)
# final_conv indica se si vuole condividere anche il primo degli ultimi due layer convoluzionali

class SplittedPPM(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6),
                 start=True, n_layers=1, final_conv=False):
        super(SplittedPPM, self).__init__()
        self.use_softmax = use_softmax
        self.start = start
        self.n_layers = n_layers
        self.final_conv = final_conv

        if start == True:
            start_layer = 0
            end_layer = start_layer + n_layers
        else:
            end_layer = len(pool_scales)
            start_layer = end_layer - n_layers

        self.ppm = []
        for i in range(start_layer, end_layer):
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_scales[i]),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        assert len(self.ppm) == n_layers
        self.ppm = nn.ModuleList(self.ppm)
        assert len(self.ppm) == n_layers

        if start == False and final_conv == False:
            # Tutto il blocco finale è nella seconda parte del decoder
            self.conv_last = nn.Sequential(
                nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                        kernel_size=3, padding=1, bias=False),
                BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(512, num_class, kernel_size=1)
            )
        elif start == True and final_conv == True:
            # Il primo layer convoluzionale è nella prima parte del decoder...
            self.conv_last = nn.Sequential(
                nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                        kernel_size=3, padding=1, bias=False),
                BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1)
            )
        elif start == False and final_conv == True:
            # ... e il secondo layer convoluzionale è nella seconda parte del decoder
            self.conv_last = nn.Sequential(nn.Conv2d(512, num_class, kernel_size=1))
        elif start == True and final_conv == False:
            self.conv_last = None

    # conv_out serve solo nel caso di decoder di tipo "end"
    # conv_out deve contenere l'output dell'encoder ed è una
    # informazione necessaria per permettere al decoder di
    # effettuare correttamente l'interpolazione
    def forward(self, x, segSize=None, conv_out=None):
        assert segSize != None

        if self.start == True:
            conv5 = x[-1]
            input_size = conv5.size()
            ppm_out = [conv5]
        elif self.start == False and self.final_conv == False:
            conv5 = conv_out[-1]
            input_size = conv_out[-1].size()
            ppm_out = x

        if self.start == True or self.final_conv == False:
            for pool_scale in self.ppm:
                ppm_out.append(nn.functional.interpolate(
                    pool_scale(conv5),
                    (input_size[2], input_size[3]),
                    mode='bilinear', align_corners=False))
        
        if self.final_conv == False:
            if self.start == True:    
                assert len(ppm_out) == self.n_layers + 1
            else:
                assert len(ppm_out) == 4 + 1

        if self.start == False and self.final_conv == False:
            ppm_out = torch.cat(ppm_out, 1)

            x = self.conv_last(ppm_out)

            x = nn.functional.interpolate(
                    x, size=segSize, mode='bilinear', align_corners=False)

            if self.use_softmax == 'softmax':
                x = nn.functional.softmax(x, dim=1)
            elif self.use_softmax == 'logsoftmax':
                x = nn.functional.log_softmax(x, dim=1)
        elif self.start == False and self.final_conv == True:
            x = self.conv_last(x)

            x = nn.functional.interpolate(
                    x, size=segSize, mode='bilinear', align_corners=False)

            if self.use_softmax == 'softmax':
                x = nn.functional.softmax(x, dim=1)
            elif self.use_softmax == 'logsoftmax':
                x = nn.functional.log_softmax(x, dim=1)
        elif self.start == True and self.final_conv == False:
            x = ppm_out
        elif self.start == True and self.final_conv == True:
            ppm_out = torch.cat(ppm_out, 1)
            x = self.conv_last(ppm_out)
            
        
        return x