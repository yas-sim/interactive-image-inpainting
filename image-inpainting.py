import sys

import cv2
import numpy as np

from openvino.inference_engine import IENetwork, IECore

# ----------------------------------------------------------------------------

g_mouseX=-1
g_mouseY=-1
g_mouseBtn = -1     # 0=left, 1=right, -1=none

g_UIState = 0       # 0: normal UI, 1: wait for a click
g_clickedFlag = False
g_inpaintFlag = False

g_penSize = 8
g_canvas = []
g_mask   = []

def clearMask():
    global g_canvas
    global g_mask
    g_mask = np.full(g_canvas.shape, [0,0,0], np.uint8)     # The size of the mask is the same as the canvas

def drawUI(image):
    cv2.circle(image, (0               , 0), 100, (   0, 255, 255), -1)
    cv2.circle(image, (image.shape[1]-1, 0), 100, (   0, 255,   0), -1)
    cv2.putText(image, 'INPAINT', (4                 ,20), cv2.FONT_HERSHEY_PLAIN, 1, (  0,   0,   0), 2)
    cv2.putText(image, 'CLEAR'  , (image.shape[1]-60 ,20), cv2.FONT_HERSHEY_PLAIN, 1, (  0,   0,   0), 2)

def drawCursor(image):
    global g_mouseX, g_mouseY
    global g_penSize
    cv2.circle(image, (g_mouseX, g_mouseY), g_penSize, (0,0,0), -1)

def dispCanvas():
    global g_canvas
    global g_mask
    canvas = g_canvas.copy()
    canvas |= g_mask
    drawUI(canvas)
    drawCursor(canvas)
    cv2.imshow('canvas', canvas)
    cv2.waitKey(1)

# Mouse event handler
def onMouse(event, x, y, flags, param):
    global g_mouseX, g_mouseY
    global g_mouseBtn
    global g_inpaintFlag
    global g_clickedFlag
    global g_UIState

    global g_mask

    black_pen = lambda x1, y1, x2, y2: cv2.line(g_mask, (x1, y1), (x2, y2), (  0,  0,  0), thickness=16)
    white_pen = lambda x1, y1, x2, y2: cv2.line(g_mask, (x1, y1), (x2, y2), (255,255,255), thickness=g_penSize)

    if g_UIState==0:      # Normal UI
        if event == cv2.EVENT_LBUTTONDOWN:
            p0=np.array([                0,                 0])
            p1=np.array([g_canvas.shape[1],                 0])
            pp=np.array([         g_mouseX,          g_mouseY])
            if np.linalg.norm(pp-p0, ord=2)<100:        # Recognition
                g_inpaintFlag = True
            elif np.linalg.norm(pp-p1, ord=2)<100:      # Clear
                clearMask()
            else:
                g_mouseBtn = 0      # left button
        if event == cv2.EVENT_LBUTTONUP:
            if g_mouseBtn==0:
                white_pen(g_mouseX, g_mouseY, x, y)
            g_mouseBtn = -1
        if event == cv2.EVENT_RBUTTONDOWN:
            g_mouseBtn = 1          # right button
        if event == cv2.EVENT_RBUTTONUP:
            if g_mouseBtn==1:
                black_pen(g_mouseX, g_mouseY, x, y)
            g_mouseBtn = -1
        if event == cv2.EVENT_MOUSEMOVE:
            if   g_mouseBtn==0:
                white_pen(g_mouseX, g_mouseY, x, y)
            elif g_mouseBtn==1:
                black_pen(g_mouseX, g_mouseY, x, y)
    elif g_UIState==1:      # no draw. wait for click state
        if event == cv2.EVENT_LBUTTONUP:
            g_clickedFlag=True

    g_mouseX = x
    g_mouseY = y

def onTrackbar(x):
    global g_penSize
    g_penSize = x

# ----------------------------------------------------------------------------

def main():
    _H=0
    _W=1
    _C=2

    global g_canvas, g_mask
    global g_threshold
    global g_UIState
    global g_inpaintFlag
    global g_clickedFlag

    if len(sys.argv)<2:
        print('Please specify an input file', file=sys.stderr)
        return -1
    g_canvas = cv2.imread(sys.argv[1])

    ie = IECore()

    model='gmcnn-places2-tf'
    model = './public/'+model+'/FP16/'+model
    net = ie.read_network(model+'.xml', model+'.bin')
    input_blob1 = 'Placeholder'
    input_blob2 = 'Placeholder_1'
    out_blob    = 'Minimum'
    in_shape1   = net.input_info[input_blob1].tensor_desc.dims  # 1,3,512,680
    in_shape2   = net.input_info[input_blob2].tensor_desc.dims
    out_shape  = net.outputs[out_blob].shape      # 1,3,512,680
    exec_net = ie.load_network(net, 'CPU')

    clearMask()
    cv2.namedWindow('canvas')
    cv2.setMouseCallback('canvas', onMouse)
    cv2.createTrackbar('Pen size', 'canvas', 8, 32, onTrackbar)

    while True:
        g_UIState = 0
        while g_inpaintFlag==False:
            dispCanvas()
            key=cv2.waitKey(100)
            if key==27:
                return
            if key==ord(' '):
                break
        g_inpaintFlag = False
        g_UIState = 1

        img = g_canvas | g_mask
        img = cv2.resize(img, (in_shape1[3], in_shape1[2]))
        img = img.transpose((_C, _H, _W))
        img = img.reshape(in_shape1)

        msk = cv2.resize(g_mask, (in_shape2[3], in_shape2[2]))
        msk = msk.transpose((_C, _H, _W))
        msk = msk[0,:,:]
        msk = np.where(msk>0., 1., 0.).astype(np.float32)
        msk = msk.reshape(in_shape2)

        res = exec_net.infer(inputs={input_blob1: img, input_blob2: msk})
        
        out = np.transpose(res[out_blob], (0, 2, 3, 1)).astype(np.uint8)

        cv2.imshow('Result', out[0])
        cv2.waitKey(1)

    return 0

if __name__ == '__main__':
    sys.exit(main())
