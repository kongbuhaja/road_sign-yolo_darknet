import numpy as np
import cv2
import imutils

dir='./test/'
def rot(img_name):
    img=cv2.imread(dir + img_name + '.jpg')
    h_i,w_i=img.shape[:2]
    cv2.imshow('img',img)

    f=open(dir + img_name + '.txt', 'r')
    line=f.readline()
    print(line)
    lbl, x, y, w, h=line.split()
    (x,y,w,h)=(float(x),float(y),float(w),float(h))

    p1=[x*w_i - w_i*w/2, y*h_i - h_i*h/2]   #left top
    p2=[p1[0]+w*w_i,p1[1]]                    #right top
    p3=[p1[0], p1[1]+h*h_i]                   #left bottom
    p4=[p2[0],p3[1]]                        #right bottom
    p=np.array([p1,p2,p3,p4])
    
    print(np.array(p1).shape)
    img2=cv2.rectangle(img.copy(),
                       (int(p1[0]), int(p1[1]),
                        int(w_i*w), int(h_i*h)),
                       (255,0,0))
    
    cv2.imshow('img2',img2)

    l=int(np.sqrt((h_i/2)**2 + (w_i/2)**2) * 2)
    print(l)
    img3=np.full((l,l,3),0,dtype='uint8')

    print(p)
    pt=p.T
    pt[0]+=(l-w_i)/2
    pt[1]+=(l-h_i)/2
    p=pt.T
    print(p)
    
    print(img.shape)
    print(img3.shape)
    for deg in range(0, 361, 30):
        img3[l//2 - h_i//2 : l//2+h_i//2, l//2 - w_i//2 : l//2 + w_i//2]=img2.copy()
        M=cv2.getRotationMatrix2D((l/2, l/2),deg,1)
        rotated=cv2.warpAffine(img3, M, (l,l))

        o=np.ones(shape=(len(p),1))
        p_o=np.hstack([p,o])
        t_p=M.dot(p_o.T).T
        t_pmin=t_p.min(0)
        t_pmax=t_p.max(0)
        cv2.rectangle(rotated,
                      (int(t_pmin[0]), int(t_pmin[1]),
                       int(t_pmax[0]-t_pmin[0]), int(t_pmax[1]-t_pmin[1])),
                      (0,0,255))
        #rotated=imutils.rotate(img3, deg)
        cv2.waitKey()
        cv2.imshow('rotated img',rotated)
    return p, t_p



a,b=rot('img')
cv2.waitKey()
cv2.destroyAllWindows()
