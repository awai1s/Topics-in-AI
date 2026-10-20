"""Offline Taxi transition reference for local smoke tests and artifact previews.

Not a substitute for official Gymnasium evaluation. Colab main.py uses Gymnasium
by default, and the PDF clearly labels which backend produced its results.
"""
from types import SimpleNamespace
import numpy as np
from PIL import Image, ImageDraw, ImageFont

MAP = [
    '+---------+',
    '|R: | : :G|',
    '| : | : : |',
    '| : : : : |',
    '| | : | : |',
    '|Y| : |B: |',
    '+---------+',
]
LOCS = ((0,0), (0,4), (4,0), (4,3))

def encode(row, col, passenger, dest):
    i = row
    i = i*5 + col
    i = i*5 + passenger
    return i*4 + dest

def decode(s):
    dest = s % 4; s //= 4
    pass_loc = s % 5; s //= 5
    col = s % 5; s //= 5
    row = s
    return row, col, pass_loc, dest

class TaxiReference:
    metadata = {'render_modes': ['rgb_array']}
    def __init__(self, render_mode=None):
        self.observation_space = SimpleNamespace(n=500)
        self.action_space = SimpleNamespace(n=6)
        self.render_mode = render_mode
        self.s = None
        self.elapsed = 0

    def reset(self, seed=None):
        rng = np.random.default_rng(seed)
        row = int(rng.integers(5)); col = int(rng.integers(5))
        passenger = int(rng.integers(4)); dst = int(rng.integers(3))
        if dst >= passenger: dst += 1
        self.s = encode(row, col, passenger, dst)
        self.elapsed = 0
        return self.s, {}

    def step(self, action):
        row, col, pas, dst = decode(self.s)
        newpas = pas
        reward = -1
        terminated = False
        if action == 0: row = min(4, row + 1)
        elif action == 1: row = max(0, row - 1)
        elif action == 2:
            if col < 4 and MAP[1+row][2*col+2] == ':': col += 1
        elif action == 3:
            if col > 0 and MAP[1+row][2*col] == ':': col -= 1
        elif action == 4:
            if pas < 4 and (row, col) == LOCS[pas]: newpas = 4
            else: reward = -10
        elif action == 5:
            if pas == 4 and (row,col) == LOCS[dst]:
                newpas = dst; terminated = True; reward = 20
            elif pas == 4 and (row,col) in LOCS:
                newpas = LOCS.index((row,col))
            else: reward = -10
        else:
            raise ValueError(action)
        self.s = encode(row, col, newpas, dst)
        self.elapsed += 1
        return self.s, reward, terminated, (self.elapsed >= 200 and not terminated), {}

    def render(self):
        im = Image.new('RGB', (470, 490), '#f6f8fb')
        dr = ImageDraw.Draw(im)
        x0, y0, cell = 42, 55, 75
        dr.text((28, 17), 'Taxi: SARSA learned policy', fill='#112944')
        row,col,pas,dst = decode(self.s)
        for r in range(5):
            for c in range(5):
                rect=(x0+c*cell,y0+r*cell,x0+(c+1)*cell,y0+(r+1)*cell)
                dr.rectangle(rect, fill='#e8eef6', outline='#bccbd9', width=1)
                if c < 4 and MAP[1+r][2*c+2] == '|':
                    x=x0+(c+1)*cell
                    dr.line((x,y0+r*cell,x,y0+(r+1)*cell), fill='#32485e', width=7)
        names=['R','G','Y','B']
        colors=['#d44839','#379959','#dda832','#4c7fbb']
        for i,(rr,cc) in enumerate(LOCS):
            xx=x0+cc*cell+8; yy=y0+rr*cell+10
            dr.rectangle((xx, yy, xx+30, yy+30), fill=colors[i])
            dr.text((xx+8,yy+8),names[i],fill='white')
        if pas<4:
            rr,cc=LOCS[pas]
            xx=x0+cc*cell+50; yy=y0+rr*cell+15
            dr.ellipse((xx-8,yy-8,xx+8,yy+8),fill='#f28500')
        xx=x0+col*cell+30; yy=y0+row*cell+46
        dr.rounded_rectangle((xx-18,yy-14,xx+24,yy+13),radius=7,fill='#ffc524',outline='#151515',width=2)
        dr.text((xx-6,yy-6),'T',fill='#151515')
        dr.text((28, 446),f'Destination: {names[dst]}   Passenger: {"onboard" if pas==4 else names[pas]}',fill='#243850')
        return np.asarray(im)

    def close(self):
        return None


def make_reference():
    return TaxiReference()
