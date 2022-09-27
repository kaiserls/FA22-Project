import cv2
import numpy as np

def render(state):
    render_state = np.asarray(state, dtype="float")
    #render_state = state
    cv2.imshow("Game", render_state)
    #cv2.resizeWindow('Game', 500, 500)
    cv2.waitKey(10)


bridge = np.zeros((50,50), dtype=bool)
cv2.namedWindow('Game',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Game', 1000, 1000)

steps = 100
episodes = 1
for i in range(0,episodes):
    for step in range(0,steps):
        print(step)
        render(bridge)
        x,y = np.random.choice(50,2)
        bridge[x,y] = 1.
