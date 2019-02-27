from __future__ import print_function, absolute_import
import time
import torch.nn as nn
import torch
from torch.autograd import Variable
from accuracy import accuracy
from utils.meters import AverageMeter
from loss import TripletLoss,FocalLoss

class BaseTrainer(object):
    def __init__(self, model, criterion):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
    
    def train(self, epoch, data_loader, optimizer, print_freq=1):
        raise NotImplementedError

    def _parse_data(self, inputs):
        imgs, ID, models, color = inputs
        inputs = [Variable(imgs)]
        model = Variable(models.cuda())
        color = Variable(color.cuda())
        ID    = Variable(ID.cuda())
        
        return inputs,model,color,ID
        #raise NotImplementedError


    def _forward(self, inputs, targets):
        raise NotImplementedError


class ResNetTrainer(BaseTrainer):
    #def _parse_data(self, inputs):
        #imgs, ID, models, color = inputs
        #inputs = [Variable(imgs)]
        #target1 = Variable(models.cuda())
        #target2 = Variable(color.cuda())
        #target3 = Variable(ID.cuda())
        
        #return inputs,target1,target2,target3
    def _forward(self, inputs, targets):
        outputs = self.model(*inputs)
        crossEntropy = nn.CrossEntropyLoss().cuda()
        model_loss = crossEntropy(outputs[0], targets[0])
        model_prec = accuracy(outputs[0].data, targets[0].data)
        color_loss = crossEntropy(outputs[1], targets[1])
        color_prec = accuracy(outputs[1].data, targets[1].data)
        id_loss, id_prec = self.criterion(outputs[3], targets[2])
        #id_prec    = accuracy(outputs[2].data, targets[2].data)
        
        return model_loss,model_prec,color_loss,model_prec,id_loss,id_prec
    

    def train(self, epoch, data_loader, optimizer, print_freq=1):
        self.model.train()
        
        #color_losses  = AverageMeter()
        #model_losses  = AverageMeter()
        total_losses = AverageMeter()
        color_prec = AverageMeter()
        model_prec = AverageMeter()
        id_prec    = AverageMeter()

        for i,inputs in enumerate(data_loader):
            
            inputs, model, color, ID = self._parse_data(inputs)
            model_loss,prec1,color_loss,prec2,id_loss,prec3 = self._forward(inputs, [model, color, ID])

            total_losses.update(model_loss.data[0] + color_loss.data[0] + id_loss.data[0], model.size(0)) 
            #color_losses.update(color_loss.data[0], color.size(0))
            #losses.update(loss, targets.size(0)) 
            # --- debug ---
            #print("model size:{}".format(model.size(0)))
            #print("color size:{}".format(color.size(0)))
            # -------------
            model_prec.update(prec1, model.size(0))
            color_prec.update(prec2, color.size(0))
            id_prec.update(prec3, ID.size(0))

            optimizer.zero_grad()
            total_loss = (0.5*model_loss + 0.5*color_loss + 1.0*id_loss) / 2.0
            total_loss.backward()
            #model_loss.backward()
            #color_loss.backward()
            optimizer.step()
      
            if (i + 1) % print_freq == 0:
                print('Epoch:[{}][{}/{}]\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Model_Prec {:.2%} ({:.2%})\t'
                      'Color_Prec {:.2%} ({:.2%})\t'
                      'ID_prec {:.2%} ({:.2%})\t'
                        .format(epoch, i + 1, len(data_loader),
                        total_losses.val,  total_losses.avg,
                        model_prec.val, model_prec.avg,
                        color_prec.val, color_prec.avg,
                        id_prec.val, id_prec.avg))


class MDNetTrainer(BaseTrainer): 

    def _forward(self, inputs, targets, name):
        pass
    def train(self, epoch, data_loader, optimizer, print_frep=1):
        pass            

            
class RAJNetTrainer(BaseTrainer):
    def _forward(self, inputs, targets):
        outputs = self.model(*inputs)
        model_loss = self.criterion(outputs[0], targets[0])
        model_prec = accuracy(outputs[0].data, targets[0].data)
        color_loss = self.criterion(outputs[1], targets[1])
        color_prec = accuracy(outputs[1].data, targets[1].data)
        id_loss    = self.criterion(outputs[1], targets[1])
        id_prec    = accuracy(outputs[2].data, targets[2].data)
        triplet    = TripletLoss().cuda()
        cat_loss   = triplet(outputs[3], targets[2].data)
        
        return model_loss,model_prec,color_loss,color_prec,id_loss,id_prec,cat_loss

    def train(self, epoch, data_loader, optimizer, print_freq=1):
        self.model.train()
        
        total_losses = AverageMeter()
        color_prec = AverageMeter()
        model_prec = AverageMeter()
        id_prec    = AverageMeter()
        
        for i,inputs in enumerate(data_loader):
            inputs,model,color,ID  = self._parse_data(inputs)
            batch_size = model.size(0)
            #print("batch_size:{}".format(batch_size))
            # 1 - model
            # 2 - color
            # 3 - ID
            loss1,prec1,loss2,prec2,loss3,prec3,loss4 = self._forward(inputs,[model,color,ID])
            total_losses.update(loss1.data[0] + loss2.data[0] + loss3.data[0] + loss4.data[0], batch_size)
            
            model_prec.update(prec1, batch_size)
            color_prec.update(prec2, batch_size)
            id_prec.update(prec3, batch_size)
            optimizer.zero_grad()
            total_loss = (0.5*loss1 + 0.5*loss2 + 1.0*loss3 + 2.0*loss4) / 4.0 
            total_loss.backward()
            optimizer.step()


            if (i + 1) % print_freq == 0:
                print('Epoch:[{}][{}/{}]\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Model_Prec {:.2%} ({:.2%})\t'
                      'Color_Prec {:.2%} ({:.2%})\t'
                      'ID_prec {:.2%} ({:.2%})\t'
                        .format(epoch, i + 1, len(data_loader),
                        total_losses.val,  total_losses.avg,
                        model_prec.val, model_prec.avg,
                        color_prec.val, color_prec.avg,
                        id_prec.val, id_prec.avg))

class VGGTrainer(BaseTrainer):
    def _forward(self, inputs, targets):
        outputs = self.model(*inputs)
        loss,prec = self.criterion(outputs, targets) 
        return loss,prec

    def train(self, epoch, data_loader, optimizer, print_freq=1):
        self.model.train()
        
        total_losses = AverageMeter()
        id_prec    = AverageMeter()
        
        for i,inputs in enumerate(data_loader):
            inputs,_,_,ID  = self._parse_data(inputs)
            batch_size = ID.size(0)
            #print("batch_size:{}".format(batch_size))
            # 1 - model
            # 2 - color
            # 3 - ID
            loss,prec  = self._forward(inputs,ID)
            total_losses.update(loss.data[0], batch_size)
            
            id_prec.update(prec, batch_size)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            if (i + 1) % print_freq == 0:
                print('Epoch:[{}][{}/{}]\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'ID_prec {:.2%} ({:.2%})\t'
                        .format(epoch, i + 1, len(data_loader),
                        total_losses.val,  total_losses.avg,
                        id_prec.val, id_prec.avg))


class ADFLTrainer(BaseTrainer):
    def _forward(self, inputs, targets):
        outputs = self.model(*inputs)
        if isinstance(self.criterion,nn.CrossEntropyLoss):
            loss = self.criterion(outputs[0], targets) 
            prec = accuracy(outputs.data, targets.data)
        else:
            loss,prec = self.criterion(outputs[1], targets)
        return loss,prec

    def train(self, epoch, data_loader, optimizer, print_freq=1):
        self.model.train()
        
        total_losses = AverageMeter()
        id_prec    = AverageMeter()
        
        for i,inputs in enumerate(data_loader):
            inputs,_,_,ID  = self._parse_data(inputs)
            batch_size = ID.size(0)
            #print("batch_size:{}".format(batch_size))
            # 1 - model
            # 2 - color
            # 3 - ID
            loss,prec  = self._forward(inputs,ID)
            total_losses.update(loss.data[0], batch_size)
            
            id_prec.update(prec, batch_size)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            if (i + 1) % print_freq == 0:
                print('Epoch:[{}][{}/{}]\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'ID_prec {:.2%} ({:.2%})\t'
                        .format(epoch, i + 1, len(data_loader),
                        total_losses.val,  total_losses.avg,
                        id_prec.val, id_prec.avg))

   















