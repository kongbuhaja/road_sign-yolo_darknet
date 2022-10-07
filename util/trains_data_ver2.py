import numpy as np
import cv2
import imutils
import glob

load_dir='./img2/'
save_dir='./img/'

def rotate(img_name):
##    print('start ',img_name)
    img=cv2.imread(load_dir + img_name + '.jpg')
    h_i,w_i=img.shape[:2]
##    cv2.imshow('img',img)

    f=open(load_dir + img_name + '.txt', 'r')
    lbl, x, y, w, h=[],[],[],[],[]
    while True:
        line=f.readline()
        if not line: break
##        print(line)
        tmp=line.split()
        lbl.append(tmp[0])
        x.append(float(tmp[1]))
        y.append(float(tmp[2]))
        w.append(float(tmp[3]))
        h.append(float(tmp[4]))

    f.close()
    
    length=np.sqrt((h_i/2)**2 + (w_i/2)**2) * 2
    l=int(length)
    img2=img.copy()
    img3=np.full((l,l,3),0,dtype='uint8')
    img4=img3.copy()
    img3[int(length/2 - h_i/2) : int(length/2+h_i/2),
         int(length/2 - w_i/2) : int(length/2 + w_i/2)]=img.copy()
    

    
    p1,p2,p3,p4,p=[],[],[],[],[]
    for cnt in range(0,len(lbl)):           #박스 반복
        p1=[x[cnt]*w_i - w_i*w[cnt]/2, y[cnt]*h_i - h_i*h[cnt]/2]   #left top
        p2=[p1[0]+w[cnt]*w_i,p1[1]]                    #right top
        p3=[p1[0], p1[1]+h[cnt]*h_i]                   #left bottom
        p4=[p2[0],p3[1]]                        #right bottom
        p.append(np.array([p1,p2,p3,p4]))
    
        cv2.rectangle(img2,
                      (int(p1[0]), int(p1[1]),
                       int(w_i*w[cnt]), int(h_i*h[cnt])),
                      (255,0,0))

##    cv2.imshow('img2',img2)

    

        pt=p[cnt].T
        pt[0]+=(l-w_i)/2
        pt[1]+=(l-h_i)/2
        p[cnt]=pt.T

    img4[int(length/2 - h_i/2) : int(length/2+h_i/2),
         int(length/2 - w_i/2) : int(length/2 + w_i/2)]=img2.copy()
    
    for deg in range(0, 360, 30):           #회전 반복
        f=open(save_dir+img_name+'_'+str(deg)+'.txt', 'w')
        M=cv2.getRotationMatrix2D((l/2, l/2),deg,1)
        rotated=cv2.warpAffine(img3, M, (l,l))
        test=cv2.warpAffine(img4, M, (l,l))
        for cnt in range(0, len(lbl)):
            o=np.ones(shape=(len(p[cnt]),1))
            p[cnt]=p[cnt].reshape(4,2)
            p_o=np.hstack([p[cnt],o])
            t_p=M.dot(p_o.T).T
            t_pmin=t_p.min(0)
            t_pmax=t_p.max(0)
            t_x, t_y, t_w, t_h=t_pmin[0], t_pmin[1], t_pmax[0]-t_pmin[0], t_pmax[1]-t_pmin[1]
            cv2.rectangle(test,
                            (int(t_x), int(t_y),
                            int(t_w), int(t_h)),
                            (0,0,255))
            t_x-=t_w/2
            t_y-=t_h/2
            box_str=str(str(lbl[cnt])+' '
                        +str(t_x)+' '
                        +str(t_y)+' '
                        +str(t_w)+' '
                        +str(t_h)+'\n')
            f.write(box_str)
            
        f.close()
        cv2.imwrite(save_dir+img_name+'_'+str(deg)+'.jpg',rotated)
        fl.write('data/img/'+img_name+'_'+str(deg)+'.txt'+'\n')
        
##        cv2.imshow('rotated img',rotated)
##        cv2.imshow('rotated test',test)
##        cv2.waitKey()
##    print('finish ', img_name)
                  

def load_filelist():
    images=glob.glob(load_dir + '*.jpg')
    return images

images=load_filelist()
trans_imglist=[]
fl=open('train.txt','w')
                 
for image in images:
    _,img_name=image.split('\\')
    if image.find('jpg') != -1 :
        img_name,_=img_name.split('.jpg')
        ty='.jpg'
    else:
        img_name,_=img_name.split('.JPG')
        ty='.JPG'
        
    print(img_name)
    rotate(img_name)

fl.close()
            
print('trans finish')
##cv2.waitKey()
##cv2.destroyAllWindows()
