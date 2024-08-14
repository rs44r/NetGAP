
import re
import math
import numpy as np
from ApplicationModel import Application, Process, Message, AAM

class UseCaseLoader:
    
    def printUsecase_info(self):
        print(str(len(self.apps))+"\t apps \n")
        print(str(len(self.proc))+"\t processes \n")
        print(str(len(self.comm))+"\t messages \n")

    def printUsecase(self):
        print("apps")
        for i in range(len(self.apps)):
            print(self.apps[i])

        print("processes")
        for i in range(len(self.proc)):
            print(self.proc[i])

        print("messages")
        for i in range(len(self.comm)):
            print(self.comm[i])
    
    def __init__(self,filename):
    
        print("oppening: "+filename)
        file = open(filename, "r") 

        temp0=re.sub("\s", "", file.read()).split("communications\":",1)
        apps=temp0[0]
        temp1 = temp0[1].split(",\"processes\":",1)
        comm=temp1[0]
        proc=temp1[1]

        #trimming the apps string
        apps=re.sub("{\"name\":\"InstanceA\",\"applications\":", "", apps)
        apps=re.sub("\[", "",re.sub("\]", "", apps))
        apps=apps.split("},{")


        #trimming the processes string
        proc=re.sub("\[", "",re.sub("\]", "", proc))
        proc=proc.split("},{")

        #trimming the comms string
        comm=re.sub("\[", "",re.sub("\]", "",comm))
        comm=comm.split("},{")



        #separating into applications
        for i in range(len(apps)):
            apps[i]=re.sub("\"","",re.sub("{","",re.sub("},","",apps[i]))).split(",",1)
            apps[i][0]=int(re.sub("name:application","",apps[i][0]))-1
            apps[i][1]=re.sub("process","",re.sub("processes:","",apps[i][1]))
            apps[i].append(0)
                #applications
                    #0 id
                    #1 processes

        #separating into messages
        for i in range(len(comm)):
            comm[i]=re.sub("\"","",re.sub("{","",re.sub("}","",comm[i]))).split(",")
            comm[i][0]=int(re.sub("req1:","",comm[i][0]))
            comm[i][1]=int(re.sub("from_process:process","",comm[i][1]))-1
            comm[i][2]=int(re.sub("to_process:process","",comm[i][2]))-1
            comm[i][3]=int(re.sub("size:","",comm[i][3]))*8
            comm[i].append((comm[i][3]*(comm[i][0]/(64/60)))/1e6*8)

            #messages
                #0 frequency
                #1 source
                #2 destination
                #3 size
               

        #separating into processes
        for i in range(len(proc)):
            proc[i]=re.sub("\"","",re.sub("{","",re.sub("}","",proc[i]))).split(",",1) 
            proc[i][0]=int(re.sub("name:process","",proc[i][0]))-1 #0 pid
            proc[i][1]=math.ceil(int(re.sub("req:","",proc[i][1]))) #1 exec time
            proc[i].append(0); #2 frequency
            proc[i].append(0); #3 period
            proc[i].append(0); #4 application
            proc[i].append(0); #5 utilization (%/sec)

                #processes
                    #0 pid
                    #1 exec_time
                    #2 frequency
                    #3 period
                    #4 application
                    #5 utilization (%/sec)

        #calculating processes periods
        for i in range(len(comm)):
            proc[comm[i][1]-1][3]=1000/comm[i][0]*64/60
            proc[comm[i][1]-1][2]=comm[i][0]
            proc[comm[i][1]-1][5]=comm[i][0]/(64/60)* proc[comm[i][1]-1][1]/1000000000


        #add processes to apps
        for i in range(len(apps)):
            temp=apps[i][1].split(",")
            temp=[(int(item)-1) for item in temp]
            apps[i][1]=np.array(temp)
            for j in range(len(temp)):
                proc[int(temp[j])][4]=apps[i][0]
                apps[i][2]+=proc[int(temp[j])-1][5]

        file.close()

        self.apps=apps
        self.proc=proc
        self.comm=comm
    
        #self.printUsecase()
    
    
    
    def rf(self,n, end, start = 0):
        return [*range(start,n), *range(n+1,end)]
    
    def loadUsecase(self,baseOPRate):
        processList={}
        messageList={}
        appList={}
        
        
        for i in range(len(self.apps)):
            a=Application(self.apps[i][0],self.apps[i][1])
            appList[str(a.aID)]=a
        
        for i in range(len(self.proc)):
            p=Process(self.proc[i][0],self.proc[i][4],self.proc[i][5]*baseOPRate,self.proc[i][2])
            #p.printProcess()
            processList[str(p.pID)]=p
            
        for i in range(len(self.comm)):
            m=Message(i,self.comm[i][1],self.comm[i][2],self.comm[i][0],self.comm[i][3])
            #m.printMessage()
            processList[str(m.source)].addMessage(m)
            processList[str(m.dest)].addMessage(m)
            messageList[str(m.mID)]=m
            
        print("Load Successful: a:{} p:{} m:{}".format(len(appList),len(processList),len(messageList)))
        
        return (appList,processList, messageList)
            
        
    