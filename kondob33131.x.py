#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cat's b3313
B3313 1.0 DECOMP  --  FILES = OFF  --  PYTHON 3 + PYGAME-CE  --  60 FPS

Single-file software-rendered 3D platformer structured like an SM64 / B3313 1.0
decomp tree (types, surface collision, level scripts, mario, area, save file,
graph renderer).  Every polygon, colour, glyph and sound is generated at runtime
from code and mathematics.  No assets are loaded; progress lives in memory only
(FILES = OFF).

Not affiliated with Nintendo or the B3313 team.  No ROM geometry, textures,
audio, maps or decomp source are reproduced.

Run:   python3 "cat'sb33130.1.1.py"
"""
import math
import random
import sys
import zlib

import colorsys
from array import array
from collections import Counter
from operator import itemgetter

try:
    import pygame
except ImportError:  # pragma: no cover
    print("cat's b3313 needs pygame-ce:   pip install pygame-ce")
    sys.exit(1)

# ----------------------------------------------------------------------------
# /* include/config.h */  constants
# ----------------------------------------------------------------------------
TITLE = "cat's b3313"
VERSION = "B3313 1.0 DECOMP"
WIN_W, WIN_H = 1280, 720
RW, RH = 426, 240                 # internal (low) render resolution
SIM_DT = 1.0 / 120.0              # fixed simulation step
MAX_STEPS = 8
TAU = math.tau
HALF_PI = math.pi * 0.5

GRAVITY = 34.0
TERMINAL = -40.0
WALK_SPEED = 6.0
RUN_SPEED = 10.5
CROUCH_SPEED = 2.4
GROUND_ACC = 46.0
GROUND_DEC = 38.0
AIR_ACC = 17.0
STEP_H = 0.55
P_RADIUS = 0.42
P_HEIGHT = 1.6
JUMP_V = (12.5, 14.6, 17.2)

LX, LY, LZ = 0.36, 0.84, 0.41
_ln = math.sqrt(LX * LX + LY * LY + LZ * LZ)
LX, LY, LZ = LX / _ln, LY / _ln, LZ / _ln


# ----------------------------------------------------------------------------
# /* src/engine/math_util.c */
# ----------------------------------------------------------------------------
def seed_hash(*parts):
    return zlib.crc32(":".join(str(p) for p in parts).encode()) & 0xFFFFFFFF


def clamp(v, a, b):
    return a if v < a else b if v > b else v


def lerp(a, b, t):
    return a + (b - a) * t


def smooth(t):
    t = clamp(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def lerpc(c1, c2, t):
    return (int(c1[0] + (c2[0] - c1[0]) * t), int(c1[1] + (c2[1] - c1[1]) * t),
            int(c1[2] + (c2[2] - c1[2]) * t))


def cmul(c, k):
    return (clamp(int(c[0] * k), 0, 255), clamp(int(c[1] * k), 0, 255), clamp(int(c[2] * k), 0, 255))


def hsv(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, clamp(s, 0, 1), clamp(v, 0, 1))
    return (int(r * 255), int(g * 255), int(b * 255))


def cjit(c, rng, amt=14):
    return (clamp(c[0] + rng.randint(-amt, amt), 0, 255), clamp(c[1] + rng.randint(-amt, amt), 0, 255),
            clamp(c[2] + rng.randint(-amt, amt), 0, 255))


def shade(col, n, amb=0.6, dif=0.42):
    l = n[0] * LX + n[1] * LY + n[2] * LZ
    if l < 0:
        l = 0.0
    k = amb + dif * l
    # slight cool tint on faces pointing away from light (retro "fake GI")
    return (clamp(int(col[0] * k), 0, 255), clamp(int(col[1] * k), 0, 255),
            clamp(int(col[2] * (k + 0.04)), 0, 255))


def _h2(ix, iz, seed):
    n = (ix * 374761393 + iz * 668265263 + seed * 2246822519) & 0xFFFFFFFF
    n = ((n ^ (n >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((n ^ (n >> 16)) & 0xFFFF) / 65535.0


def vnoise(x, z, seed=0):
    ix = math.floor(x)
    iz = math.floor(z)
    fx = x - ix
    fz = z - iz
    u = fx * fx * (3 - 2 * fx)
    v = fz * fz * (3 - 2 * fz)
    a = _h2(ix, iz, seed)
    b = _h2(ix + 1, iz, seed)
    c = _h2(ix, iz + 1, seed)
    d = _h2(ix + 1, iz + 1, seed)
    return (a + (b - a) * u) + ((c + (d - c) * u) - (a + (b - a) * u)) * v


def fbm(x, z, seed=0, octaves=3):
    s = 0.0
    amp = 1.0
    tot = 0.0
    for o in range(octaves):
        s += vnoise(x, z, seed + o * 101) * amp
        tot += amp
        x *= 2.03
        z *= 2.03
        amp *= 0.5
    return s / tot


def newell(pts):
    nx = ny = nz = 0.0
    L = len(pts)
    for i in range(L):
        a = pts[i]
        b = pts[(i + 1) % L]
        nx += (a[1] - b[1]) * (a[2] + b[2])
        ny += (a[2] - b[2]) * (a[0] + b[0])
        nz += (a[0] - b[0]) * (a[1] + b[1])
    ln = math.sqrt(nx * nx + ny * ny + nz * nz)
    if ln < 1e-9:
        return None
    return (nx / ln, ny / ln, nz / ln)


class Vec3f:  # /* include/types.h */
    """Vec3f — SM64-style 3D vector (renderer uses raw tuples for speed)."""
    __slots__ = ("x", "y", "z")

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __add__(self, o):
        return Vec3f(self.x + o.x, self.y + o.y, self.z + o.z)

    def __sub__(self, o):
        return Vec3f(self.x - o.x, self.y - o.y, self.z - o.z)

    def __mul__(self, k):
        return Vec3f(self.x * k, self.y * k, self.z * k)

    __rmul__ = __mul__

    def dot(self, o):
        return self.x * o.x + self.y * o.y + self.z * o.z

    def cross(self, o):
        return Vec3f(self.y * o.z - self.z * o.y, self.z * o.x - self.x * o.z, self.x * o.y - self.y * o.x)

    def length(self):
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def hlen(self):
        return math.hypot(self.x, self.z)

    def normalized(self):
        l = self.length()
        return Vec3f(self.x / l, self.y / l, self.z / l) if l > 1e-9 else Vec3f()

    def copy(self):
        return Vec3f(self.x, self.y, self.z)

    def set(self, x, y, z):
        self.x, self.y, self.z = x, y, z
        return self

    def tup(self):
        return (self.x, self.y, self.z)

    def __repr__(self):
        return "Vec3f(%.2f, %.2f, %.2f)" % (self.x, self.y, self.z)


# ----------------------------------------------------------------------------
# /* src/game/hud_font.c */  built-in 5x7 bitmap font (FILES = OFF — no font files)
# ----------------------------------------------------------------------------
_G = {
    'A': (14, 17, 17, 31, 17, 17, 17), 'B': (30, 17, 17, 30, 17, 17, 30), 'C': (14, 17, 16, 16, 16, 17, 14),
    'D': (30, 17, 17, 17, 17, 17, 30), 'E': (31, 16, 16, 30, 16, 16, 31), 'F': (31, 16, 16, 30, 16, 16, 16),
    'G': (14, 17, 16, 23, 17, 17, 15), 'H': (17, 17, 17, 31, 17, 17, 17), 'I': (14, 4, 4, 4, 4, 4, 14),
    'J': (7, 2, 2, 2, 2, 18, 12), 'K': (17, 18, 20, 24, 20, 18, 17), 'L': (16, 16, 16, 16, 16, 16, 31),
    'M': (17, 27, 21, 21, 17, 17, 17), 'N': (17, 17, 25, 21, 19, 17, 17), 'O': (14, 17, 17, 17, 17, 17, 14),
    'P': (30, 17, 17, 30, 16, 16, 16), 'Q': (14, 17, 17, 17, 21, 18, 13), 'R': (30, 17, 17, 30, 20, 18, 17),
    'S': (15, 16, 16, 14, 1, 1, 30), 'T': (31, 4, 4, 4, 4, 4, 4), 'U': (17, 17, 17, 17, 17, 17, 14),
    'V': (17, 17, 17, 17, 17, 10, 4), 'W': (17, 17, 17, 21, 21, 21, 10), 'X': (17, 17, 10, 4, 10, 17, 17),
    'Y': (17, 17, 10, 4, 4, 4, 4), 'Z': (31, 1, 2, 4, 8, 16, 31),
    '0': (14, 17, 19, 21, 25, 17, 14), '1': (4, 12, 4, 4, 4, 4, 14), '2': (14, 17, 1, 2, 4, 8, 31),
    '3': (31, 2, 4, 2, 1, 17, 14), '4': (2, 6, 10, 18, 31, 2, 2), '5': (31, 16, 30, 1, 1, 17, 14),
    '6': (6, 8, 16, 30, 17, 17, 14), '7': (31, 1, 2, 4, 8, 8, 8), '8': (14, 17, 17, 14, 17, 17, 14),
    '9': (14, 17, 17, 15, 1, 2, 12), ' ': (0, 0, 0, 0, 0, 0, 0), '.': (0, 0, 0, 0, 0, 12, 12),
    ',': (0, 0, 0, 0, 12, 4, 8), ':': (0, 12, 12, 0, 12, 12, 0), '!': (4, 4, 4, 4, 4, 0, 4),
    '?': (14, 17, 1, 2, 4, 0, 4), '-': (0, 0, 0, 31, 0, 0, 0), '+': (0, 4, 4, 31, 4, 4, 0),
    '/': (1, 1, 2, 4, 8, 16, 16), "'": (4, 4, 8, 0, 0, 0, 0), '"': (10, 10, 0, 0, 0, 0, 0),
    '(': (2, 4, 8, 8, 8, 4, 2), ')': (8, 4, 2, 2, 2, 4, 8), '=': (0, 0, 31, 0, 31, 0, 0),
    '>': (8, 4, 2, 1, 2, 4, 8), '<': (2, 4, 8, 16, 8, 4, 2), '#': (10, 10, 31, 10, 31, 10, 10),
    '%': (24, 25, 2, 4, 8, 19, 3), '*': (4, 21, 14, 31, 14, 21, 4), '[': (14, 8, 8, 8, 8, 8, 14),
    ']': (14, 2, 2, 2, 2, 2, 14), '_': (0, 0, 0, 0, 0, 0, 31), '&': (12, 18, 20, 8, 21, 18, 13),
    '~': (0, 0, 8, 21, 2, 0, 0), '^': (4, 10, 17, 0, 0, 0, 0), '|': (4, 4, 4, 4, 4, 4, 4),
    '@': (14, 17, 23, 21, 23, 16, 14), '$': (4, 15, 20, 14, 5, 30, 4), ';': (0, 12, 12, 0, 12, 4, 8),
    '{': (6, 4, 4, 8, 4, 4, 6), '}': (12, 4, 4, 2, 4, 4, 12),     '\x01': (14, 31, 31, 31, 14, 4, 14),  # spaceworld mario head (lives)
    '\x02': (4, 4, 14, 31, 14, 4, 4),  # dream shard / star glyph
    '\x03': (14, 17, 17, 14, 4, 14, 31),  # spaceworld coin
}


class Font:
    def __init__(self):
        self.cache = {}

    def width(self, text, scale=1):
        return max(0, len(text) * 6 * scale - scale)

    def render(self, text, col=(255, 255, 255), scale=1, shadow=True):
        key = (text, col, scale, shadow)
        s = self.cache.get(key)
        if s is not None:
            return s
        if len(self.cache) > 900:
            self.cache.clear()
        w = self.width(text, scale) + (scale if shadow else 0) + 1
        h = 7 * scale + (scale if shadow else 0) + 1
        s = pygame.Surface((max(1, w), h), pygame.SRCALPHA)
        up = text.upper()
        passes = ((scale, (0, 0, 0)), (0, col)) if shadow else ((0, col),)
        for off, c in passes:
            x = 0
            for ch in up:
                g = _G.get(ch, (31, 17, 17, 17, 17, 17, 31))
                for ry, row in enumerate(g):
                    if row:
                        for rx in range(5):
                            if row & (16 >> rx):
                                s.fill(c, (x + rx * scale + off, ry * scale + off, scale, scale))
                x += 6 * scale
        self.cache[key] = s
        return s

    def draw(self, surf, text, x, y, col=(255, 255, 255), scale=1, center=False, shadow=True, right=False):
        s = self.render(text, col, scale, shadow)
        if center:
            x -= s.get_width() // 2
        elif right:
            x -= s.get_width()
        surf.blit(s, (int(x), int(y)))
        return s.get_width()

    def wrap(self, text, maxchars):
        words = text.split(" ")
        lines = []
        cur = ""
        for w in words:
            if len(cur) + len(w) + (1 if cur else 0) > maxchars:
                lines.append(cur)
                cur = w
            else:
                cur = (cur + " " + w) if cur else w
        if cur:
            lines.append(cur)
        return lines


# ----------------------------------------------------------------------------
# /* src/audio/external.c */  procedural audio (synthesized; silent on failure)
# ----------------------------------------------------------------------------
class Audio:
    def __init__(self):
        self.ok = False
        self.sfx_on = True
        self.music_on = True
        self.sounds = {}
        self.tracks = {}
        self.cur = None
        self.chan = None
        self.rate = 22050
        self.channels = 2
        try:
            try:
                pygame.mixer.init(22050, -16, 2, 512, allowedchanges=0)
            except TypeError:
                pygame.mixer.init(22050, -16, 2, 512)
            got = pygame.mixer.get_init()
            if not got or got[1] != -16:
                raise RuntimeError("unsupported mixer format")
            self.rate, _, self.channels = got
            pygame.mixer.set_num_channels(20)
            self.ok = True
        except Exception:
            self.ok = False

    # -- synthesis primitives --------------------------------------------------
    def _make(self, samples):
        a = array('h', [int(32000 * (1.0 if s > 1.0 else -1.0 if s < -1.0 else s)) for s in samples])
        if self.channels == 2:
            b = array('h', bytes(4 * len(a)))
            b[0::2] = a
            b[1::2] = a
            a = b
        return pygame.mixer.Sound(buffer=a.tobytes())

    def _sweep(self, f0, f1, dur, wave=0, vol=0.3, curve=1.0, decay=1.5, noise=0.0, vib=0.0, seed=1):
        r = self.rate
        n = max(1, int(r * dur))
        out = [0.0] * n
        ph = 0.0
        rnd = random.Random(seed)
        att = max(1, int(0.004 * r))
        for i in range(n):
            t = i / n
            f = f0 + (f1 - f0) * (t ** curve)
            if vib:
                f *= 1.0 + vib * math.sin(i / r * 38.0)
            ph += f / r
            p = ph - int(ph)
            if wave == 0:
                v = 1.0 if p < 0.5 else -1.0
            elif wave == 1:
                v = 4.0 * abs(p - 0.5) - 1.0
            elif wave == 2:
                v = math.sin(TAU * p)
            else:
                v = 2.0 * p - 1.0
            if noise:
                v = v * (1 - noise) + (rnd.random() * 2 - 1) * noise
            env = (1 - t) ** decay
            if i < att:
                env *= i / att
            out[i] = v * env * vol
        return out

    def _seq(self, freqs, each, wave=1, vol=0.3, decay=1.2, vib=0.0, tail=0.0):
        out = []
        for i, f in enumerate(freqs):
            d = each + (tail if i == len(freqs) - 1 else 0.0)
            out += self._sweep(f, f, d, wave, vol, decay=decay, vib=vib)
        return out

    @staticmethod
    def _mix(*parts):
        n = max(len(p) for p in parts)
        out = [0.0] * n
        for p in parts:
            for i, v in enumerate(p):
                out[i] += v
        return out

    # -- build everything (generator: yields progress for the loading screen) -
    def build(self, seed):
        if not self.ok:
            yield 1.0
            return
        S = {}
        S['jump'] = self._sweep(260, 720, 0.14, 0, 0.18, 0.7)
        S['jump2'] = self._sweep(330, 900, 0.16, 0, 0.18, 0.7)
        S['jump3'] = self._sweep(300, 1250, 0.24, 1, 0.3, 0.6, vib=0.05)
        S['flip'] = self._sweep(200, 1100, 0.32, 1, 0.3, 1.3, vib=0.08)
        S['long'] = self._sweep(500, 260, 0.22, 0, 0.14, 1.0, noise=0.25)
        S['star'] = self._seq([784, 988, 1175, 1568, 2093], 0.07, 1, 0.3, tail=0.35)
        S['coin'] = self._sweep(1300, 1750, 0.08, 0, 0.1)
        S['portal'] = self._mix(self._sweep(900, 110, 0.75, 2, 0.35, 0.6, 1.0, vib=0.1),
                                self._sweep(300, 300, 0.75, 2, 0.12, decay=0.8, noise=0.9, seed=3))
        S['damage'] = self._sweep(420, 80, 0.32, 3, 0.3, 0.8, noise=0.25)
        S['menu'] = self._sweep(880, 880, 0.04, 0, 0.1)
        S['select'] = self._sweep(600, 1250, 0.09, 0, 0.12)
        S['enemy'] = self._sweep(520, 60, 0.2, 0, 0.22, 0.8, noise=0.3)
        S['switch'] = self._seq([440, 660, 880], 0.06, 0, 0.14)
        S['secret'] = self._seq([523, 622, 784, 932, 1046, 1245], 0.09, 1, 0.3, vib=0.03, tail=0.4)
        S['talk'] = self._sweep(700, 640, 0.035, 0, 0.08)
        S['pound'] = self._sweep(160, 40, 0.26, 3, 0.35, 0.5, noise=0.45)
        S['lock'] = self._seq([196, 147], 0.08, 0, 0.14)
        S['life'] = self._seq([523, 659, 784, 1046, 784, 1046], 0.07, 0, 0.16, tail=0.2)
        S['splash'] = self._sweep(900, 200, 0.3, 2, 0.12, decay=1.0, noise=0.85, seed=9)
        S['ending'] = self._seq([392, 523, 659, 784, 1046, 1318, 1568], 0.16, 1, 0.3, vib=0.02, tail=1.2)
        for k, v in S.items():
            self.sounds[k] = self._make(v)
        yield 0.25
        for i, mood in enumerate(('calm', 'eerie', 'dream')):
            self.tracks[mood] = self._make(self._music(mood, seed + i))
            yield 0.25 + 0.25 * (i + 1)

    def _music(self, mood, seed):
        r = self.rate
        rnd = random.Random(seed * 7 + 11)
        TAB = [math.sin(TAU * i / 2048) for i in range(2048)]
        if mood == 'calm':
            root, bar, scale, prog, qual, step = 196.0, 2.0, [0, 2, 4, 7, 9], [0, 9, 5, 7], [0, 1, 0, 0], 0.25
        elif mood == 'eerie':
            root, bar, scale, prog, qual, step = 146.8, 2.6, [0, 1, 5, 7, 8], [0, 1, -4, 6], [1, 0, 1, 2], 0.325
        else:
            root, bar, scale, prog, qual, step = 174.6, 2.4, [0, 2, 4, 6, 8, 10], [0, 2, 8, 10], [3, 3, 0, 2], 0.3
        chords = {0: (0, 4, 7), 1: (0, 3, 7), 2: (0, 3, 6), 3: (0, 4, 11)}
        bars = 8
        n = int(bars * bar * r)
        buf = [0.0] * n
        # pad
        for b in range(bars):
            deg = prog[b % len(prog)]
            ivs = chords[qual[b % len(qual)]]
            fr = [root * 0.5 * 2 ** ((deg + iv) / 12.0) for iv in ivs]
            s0 = int(b * bar * r)
            L = int(bar * r)
            phs = [0.0, 0.0, 0.0]
            incs = [f * 2048 / r for f in fr]
            fade = int(0.35 * r)
            for i in range(L):
                e = 1.0
                if i < fade:
                    e = i / fade
                elif i > L - fade:
                    e = (L - i) / fade
                v = 0.0
                for k in range(3):
                    phs[k] += incs[k]
                    v += TAB[int(phs[k]) & 2047]
                buf[s0 + i] += v * 0.055 * e
        # melody + bass
        steps = int(bar / step)
        mel_dur = int(step * r * 2.2)
        last = 0
        for b in range(bars):
            deg = prog[b % len(prog)]
            for st in range(steps):
                t0 = int((b * bar + st * step) * r)
                if st % max(1, steps // 2) == 0:
                    f = root * 0.25 * 2 ** (deg / 12.0)
                    L = int(step * r * 1.6)
                    ph = 0.0
                    for i in range(min(L, n - t0)):
                        ph += f / r
                        v = 1.0 if (ph - int(ph)) < 0.5 else -1.0
                        buf[t0 + i] += v * 0.035 * (1 - i / L)
                if rnd.random() < (0.42 if mood == 'calm' else 0.3):
                    last = clamp(last + rnd.choice((-2, -1, 1, 2, 0)), 0, len(scale) * 2 - 1)
                    semi = scale[last % len(scale)] + 12 * (last // len(scale)) + deg * (mood == 'calm')
                    f = root * 2 * 2 ** (semi / 12.0)
                    ph = 0.0
                    for i in range(min(mel_dur, n - t0)):
                        ph += f / r
                        p = ph - int(ph)
                        v = 4.0 * abs(p - 0.5) - 1.0
                        buf[t0 + i] += v * 0.09 * math.exp(-i / (0.12 * r))
        # echo
        d = int(step * 1.5 * r)
        for i in range(d, n):
            buf[i] += buf[i - d] * 0.33
        m = max(1e-6, max(abs(v) for v in buf))
        g = 0.55 / m
        return [v * g for v in buf]

    # -- playback ---------------------------------------------------------------
    def play(self, name, vol=1.0):
        if not (self.ok and self.sfx_on):
            return
        s = self.sounds.get(name)
        if s:
            try:
                ch = s.play()
                if ch:
                    ch.set_volume(vol)
            except Exception:
                pass

    def music(self, mood):
        if not self.ok:
            return
        if not self.music_on:
            self.stop_music()
            return
        if mood == self.cur and self.chan is not None and self.chan.get_busy():
            return
        self.stop_music()
        t = self.tracks.get(mood)
        if t:
            try:
                self.chan = t.play(loops=-1, fade_ms=900)
                if self.chan:
                    self.chan.set_volume(0.38)
                self.cur = mood
            except Exception:
                self.chan = None

    def stop_music(self):
        if self.chan is not None:
            try:
                self.chan.fadeout(500)
            except Exception:
                pass
        self.chan = None
        self.cur = None


# ----------------------------------------------------------------------------
# /* src/engine/surface_collision.c */
# ----------------------------------------------------------------------------
class Surface:  # /* SURFACE_* collision */
    """Collision primitive.  k: 0 box, 1 ramp, 2 cylinder, 3 heightfield, 4 ring wall (inside of a cylinder)."""
    __slots__ = ("k", "x0", "y0", "z0", "x1", "y1", "z1", "a", "b", "axis", "cx", "cz", "r",
                 "H", "cell", "nx", "nz", "stamp", "tag", "mover")

    def __init__(self, k):
        self.k = k
        self.stamp = 0
        self.tag = None
        self.mover = None
        self.x0 = self.y0 = self.z0 = self.x1 = self.y1 = self.z1 = 0.0
        self.a = self.b = 0.0
        self.axis = 0
        self.cx = self.cz = self.r = 0.0
        self.H = None
        self.cell = 1.0
        self.nx = self.nz = 0

    def top(self, x, z):
        k = self.k
        if k == 1:
            if self.axis == 0:
                t = (x - self.x0) / (self.x1 - self.x0)
            else:
                t = (z - self.z0) / (self.z1 - self.z0)
            t = 0.0 if t < 0 else 1.0 if t > 1 else t
            return self.a + (self.b - self.a) * t
        if k == 3:
            fx = (x - self.x0) / self.cell
            fz = (z - self.z0) / self.cell
            i = int(fx)
            j = int(fz)
            if i < 0:
                i = 0
            elif i >= self.nx:
                i = self.nx - 1
            if j < 0:
                j = 0
            elif j >= self.nz:
                j = self.nz - 1
            u = clamp(fx - i, 0.0, 1.0)
            v = clamp(fz - j, 0.0, 1.0)
            H = self.H
            h00 = H[j][i]
            h10 = H[j][i + 1]
            h11 = H[j + 1][i + 1]
            h01 = H[j + 1][i]
            if u >= v:
                return h00 + (h10 - h00) * u + (h11 - h10) * v
            return h00 + (h11 - h01) * u + (h01 - h00) * v
        return self.y1

    def contains(self, x, z, m=0.0):
        k = self.k
        if k == 2:
            dx = x - self.cx
            dz = z - self.cz
            rr = self.r + m
            return dx * dx + dz * dz <= rr * rr
        if k == 4:
            return False
        return self.x0 - m <= x <= self.x1 + m and self.z0 - m <= z <= self.z1 + m

    def solid_at(self, x, y, z):
        k = self.k
        if k == 4:
            dx = x - self.cx
            dz = z - self.cz
            return self.y0 <= y <= self.y1 and dx * dx + dz * dz > self.r * self.r
        if not self.contains(x, z):
            return False
        if k == 3:
            return y < self.top(x, z)
        return self.y0 <= y <= self.top(x, z)


class MovingPlatform:
    """Moving platform (elevators, gears, pistons...)."""

    def __init__(self, col, hx, hy, hz, base, kind, amp, speed, phase, color):
        self.col = col
        col.mover = self
        self.hx, self.hy, self.hz = hx, hy, hz
        self.base = base
        self.kind = kind
        self.amp = amp
        self.speed = speed
        self.phase = phase
        self.color = color
        self.x, self.y, self.z = base
        self.dx = self.dy = self.dz = 0.0
        self.place(0.0)
        self.dx = self.dy = self.dz = 0.0

    def place(self, t):
        bx, by, bz = self.base
        if self.kind == 'circle':
            a = t * self.speed + self.phase
            nx, ny, nz = bx + math.cos(a) * self.amp[0], by, bz + math.sin(a) * self.amp[0]
        else:
            s = math.sin(t * self.speed + self.phase)
            nx, ny, nz = bx + self.amp[0] * s, by + self.amp[1] * s, bz + self.amp[2] * s
        self.dx, self.dy, self.dz = nx - self.x, ny - self.y, nz - self.z
        self.x, self.y, self.z = nx, ny, nz
        c = self.col
        c.x0, c.x1 = nx - self.hx, nx + self.hx
        c.y0, c.y1 = ny - self.hy, ny + self.hy
        c.z0, c.z1 = nz - self.hz, nz + self.hz


# ----------------------------------------------------------------------------
# /* src/game/object_helpers.h */  WarpNode / StarDef (B3313 1.0)
# ----------------------------------------------------------------------------
class WarpNode:  # /* warps / painting warps */
    def __init__(self, aid, v, name, kind, x, y, z, yaw, w, h, rules, lock, visible, fy, color):
        self.aid, self.v, self.name, self.kind = aid, v, name, kind
        self.pid = "%s:%d:%s" % (aid, v, name)
        self.x, self.y, self.z, self.yaw, self.w, self.h = x, y, z, yaw, w, h
        self.fx, self.fz = math.sin(yaw), math.cos(yaw)
        self.rules = rules
        self.lock = lock
        self.visible = visible
        self.fy = fy
        self.color = color
        self.armed = True

    def dest(self, w):
        for cond, d in self.rules:
            if cond is None or cond(w):
                return d
        return None

    def is_locked(self, w):
        return self.lock is not None and not self.lock[0](w)

    def is_visible(self, w):
        return self.visible is None or self.visible(w)

    def contains(self, px, py, pz):
        dx = px - self.x
        dz = pz - self.z
        k = self.kind
        if k == 'hole':
            hw = self.w * 0.5
            return abs(dx) < hw and abs(dz) < hw and self.y - 2.5 < py < self.y + 0.3
        if k == 'pipe':
            rr = self.w * 0.42
            return dx * dx + dz * dz < rr * rr and abs(py - self.y) < 0.4
        lat = dx * self.fz - dz * self.fx
        dep = dx * self.fx + dz * self.fz
        if abs(lat) > self.w * 0.5:
            return False
        if k == 'painting':
            return -0.3 < dep < 0.8 and py + P_HEIGHT > self.y + 0.3 and py < self.y + self.h - 0.3
        if k == 'ring':
            m = py + 0.8
            return abs(dep) < 0.7 and self.y - self.h * 0.5 < m < self.y + self.h * 0.5
        return -0.3 < dep < 0.6 and py < self.y + self.h - 0.3 and py + P_HEIGHT > self.y

    def exit_point(self):
        d = {'door': 3.0, 'painting': 2.4, 'ring': 2.0}.get(self.kind, 2.2)
        return (self.x + self.fx * d, self.fy, self.z + self.fz * d)


class StarDef:  # /* Power Star spawn */
    __slots__ = ("sid", "x", "y", "z", "cond", "final", "label")

    def __init__(self, sid, x, y, z, cond, final, label):
        self.sid, self.x, self.y, self.z, self.cond, self.final, self.label = sid, x, y, z, cond, final, label


# ----------------------------------------------------------------------------
# /* src/goddard / gfx */  static mesh chunk
# ----------------------------------------------------------------------------
class GfxMesh:
    __slots__ = ("verts", "faces", "cx", "cy", "cz", "rad")

    def __init__(self):
        self.verts = []
        self.faces = []
        self.cx = self.cy = self.cz = self.rad = 0.0


def _bil(p0, p1, p2, p3, u, v):
    ax = p0[0] + (p1[0] - p0[0]) * u
    ay = p0[1] + (p1[1] - p0[1]) * u
    az = p0[2] + (p1[2] - p0[2]) * u
    bx = p3[0] + (p2[0] - p3[0]) * u
    by = p3[1] + (p2[1] - p3[1]) * u
    bz = p3[2] + (p2[2] - p3[2]) * u
    return (ax + (bx - ax) * v, ay + (by - ay) * v, az + (bz - az) * v)


# ----------------------------------------------------------------------------
# /* src/game/area.h */  Area — playable location
# ----------------------------------------------------------------------------
class Area:  # /* src/game/area.h */
    CELL = 8.0
    _stamp = 0

    def __init__(self):
        self.meshes = []
        self.cols = []
        self.grid = {}
        self.big = []
        self.movers = []
        self.portals = []
        self.stars = []
        self.coins = []
        self.npcs = []
        self.enemies = []
        self.switches = []
        self.triggers = []
        self.spawns = {}
        self.nfaces = 0

    def index(self):
        C = self.CELL
        g = self.grid
        for c in self.cols:
            if c.k in (3, 4):
                self.big.append(c)
                continue
            if c.k == 2:
                x0, x1, z0, z1 = c.cx - c.r, c.cx + c.r, c.cz - c.r, c.cz + c.r
            else:
                x0, x1, z0, z1 = c.x0, c.x1, c.z0, c.z1
            for ix in range(int(math.floor(x0 / C)), int(math.floor(x1 / C)) + 1):
                for iz in range(int(math.floor(z0 / C)), int(math.floor(z1 / C)) + 1):
                    g.setdefault((ix, iz), []).append(c)

    def query(self, x, z, r):
        Area._stamp += 1
        st = Area._stamp
        C = self.CELL
        out = []
        g = self.grid
        for ix in range(int(math.floor((x - r) / C)), int(math.floor((x + r) / C)) + 1):
            for iz in range(int(math.floor((z - r) / C)), int(math.floor((z + r) / C)) + 1):
                lst = g.get((ix, iz))
                if lst:
                    for c in lst:
                        if c.stamp != st:
                            c.stamp = st
                            out.append(c)
        out.extend(self.big)
        for m in self.movers:
            out.append(m.col)
        return out

    def ground(self, x, z, ymax, m=0.0, cols=None):
        best = -1e9
        bc = None
        if cols is None:
            cols = self.query(x, z, 0.5)
        for c in cols:
            if c.k == 4 or not c.contains(x, z, m):
                continue
            t = c.top(x, z)
            if t <= ymax and t > best:
                best = t
                bc = c
        return best, bc

    def point_solid(self, x, y, z):
        for c in self.query(x, z, 0.3):
            if c.solid_at(x, y, z):
                return True
        return False

    def cam_solid(self, x, y, z):
        """Camera occlusion test: solids plus the thin planes of doors (so the camera never peeks through them)."""
        for p in self.portals:
            if p.kind == 'door':
                dx, dz = x - p.x, z - p.z
                dep = dx * p.fx + dz * p.fz
                if -2.0 < dep < 0.15 and abs(dx * p.fz - dz * p.fx) < p.w * 0.5 + 0.6 and p.y - 0.5 < y < p.y + p.h + 0.8:
                    return True
        return self.point_solid(x, y, z)

    def update_movers(self, t):
        for m in self.movers:
            m.place(t)


# ----------------------------------------------------------------------------
# /* levels/*/script.c */  LevelScript — B3313 1.0 procedural level build
# ----------------------------------------------------------------------------
class LevelScript:  # /* levels/*/script.c */
    def __init__(self, world, aid, v, dry=False):
        self.w = world
        self.aid = aid
        self.v = v
        self.dry = dry
        self.rng = random.Random(seed_hash(world.seed, aid, v))
        self.faces = []
        self.cols = []
        self.movers = []
        self.portals = []
        self.power_stars = []
        self.coins = []
        self.npcs = []
        self.enemies = []
        self.switches = []
        self.triggers = []
        self.spawns = {}
        self.s = 1.0
        self.ox = self.oy = self.oz = 0.0
        self.mx = 1
        self.amb = 0.6
        self.dif = 0.42
        self.sky = ((60, 110, 230), (190, 220, 255), (90, 110, 90))
        self.fog = (190, 215, 250)
        self.water = None
        self.water_col = (40, 90, 190)
        self.lava = None
        self.kill_y = -45.0
        self.starfield = False
        self.wobble = False
        self.hue_cycle = False
        self.view = 260.0
        self.fog_near = 0.35
        self.glitch = False
        self.music = None
        self.nstar = 0
        self.suppress = False    # replica mode: geometry only

    # -- transforms ----------------------------------------------------------
    def P(self, x, y, z):
        s = self.s
        return (self.ox + self.mx * s * x, self.oy + s * y, self.oz + s * z)

    def Y(self, yaw):
        return -yaw if self.mx < 0 else yaw

    # -- faces ------------------------------------------------------------------
    def face(self, pts, col, hint, dbl=False, raw=False, bias=0.0):
        if self.dry:
            return
        s = self.s
        mx = self.mx * s
        ox, oy, oz = self.ox, self.oy, self.oz
        tp = [(ox + mx * p[0], oy + s * p[1], oz + s * p[2]) for p in pts]
        n = newell(tp)
        h = (hint[0] * self.mx, hint[1], hint[2])
        if n is None:
            l = math.sqrt(h[0] * h[0] + h[1] * h[1] + h[2] * h[2]) or 1.0
            n = (h[0] / l, h[1] / l, h[2] / l)
        elif n[0] * h[0] + n[1] * h[1] + n[2] * h[2] < 0:
            n = (-n[0], -n[1], -n[2])
        c = col if raw else shade(col, n, self.amb, self.dif)
        self.faces.append((tp, c, n, dbl, bias))

    def qgrid(self, p0, p1, p2, p3, nu, nv, col, hint, col2=None, dbl=False, raw=False, bias=0.0):
        if self.dry:
            return
        for j in range(nv):
            v0 = j / nv
            v1 = (j + 1) / nv
            for i in range(nu):
                u0 = i / nu
                u1 = (i + 1) / nu
                q = [_bil(p0, p1, p2, p3, u0, v0), _bil(p0, p1, p2, p3, u1, v0),
                     _bil(p0, p1, p2, p3, u1, v1), _bil(p0, p1, p2, p3, u0, v1)]
                c = col2 if (col2 is not None and (i + j) & 1) else col
                self.face(q, c, hint, dbl, raw, bias)

    # -- colliders ----------------------------------------------------------
    def col_box(self, x0, y0, z0, x1, y1, z1, tag=None):
        if self.dry:
            return None
        a = self.P(x0, y0, z0)
        b = self.P(x1, y1, z1)
        c = Surface(0)
        c.x0, c.x1 = min(a[0], b[0]), max(a[0], b[0])
        c.y0, c.y1 = min(a[1], b[1]), max(a[1], b[1])
        c.z0, c.z1 = min(a[2], b[2]), max(a[2], b[2])
        c.tag = tag
        self.cols.append(c)
        return c

    # -- primitives -------------------------------------------------------------
    def box(self, x0, y0, z0, x1, y1, z1, col, top=None, solid=True, visible=True, tile=6.0,
            checker=None, bottom=True, raw=False, bias=0.0, tag=None):
        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0
        if z0 > z1:
            z0, z1 = z1, z0
        if visible and not self.dry:
            nx = max(1, int((x1 - x0) / tile + 0.5))
            ny = max(1, int((y1 - y0) / tile + 0.5))
            nz = max(1, int((z1 - z0) / tile + 0.5))
            tc = top if top is not None else col
            self.qgrid((x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1), nx, nz, tc, (0, 1, 0), checker,
                       raw=raw, bias=bias)
            if bottom:
                self.qgrid((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1), nx, nz, col, (0, -1, 0),
                           raw=raw, bias=bias)
            self.qgrid((x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), nx, ny, col, (0, 0, -1), raw=raw,
                       bias=bias)
            self.qgrid((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1), nx, ny, col, (0, 0, 1), raw=raw,
                       bias=bias)
            self.qgrid((x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0), nz, ny, col, (-1, 0, 0), raw=raw,
                       bias=bias)
            self.qgrid((x1, y0, z0), (x1, y0, z1), (x1, y1, z1), (x1, y1, z0), nz, ny, col, (1, 0, 0), raw=raw,
                       bias=bias)
        if solid:
            return self.col_box(x0, y0, z0, x1, y1, z1, tag)
        return None

    def ramp(self, x0, z0, x1, z1, y0, a, b, axis, col, top=None, solid=True):
        """Sloped block. axis 0: height a at x0 -> b at x1.  axis 1: a at z0 -> b at z1."""
        if axis == 0:
            h00, h10, h11, h01 = a, b, b, a
        else:
            h00, h10, h11, h01 = a, a, b, b
        if not self.dry:
            L = (x1 - x0) if axis == 0 else (z1 - z0)
            n = max(1, int(L / 5 + 0.5))
            W = (z1 - z0) if axis == 0 else (x1 - x0)
            m = max(1, int(W / 6 + 0.5))
            tc = top or col
            if axis == 0:
                self.qgrid((x0, h00, z0), (x1, h10, z0), (x1, h11, z1), (x0, h01, z1), n, m, tc, (0, 1, 0))
            else:
                self.qgrid((x0, h00, z0), (x1, h10, z0), (x1, h11, z1), (x0, h01, z1), m, n, tc, (0, 1, 0))
            self.face([(x0, y0, z0), (x1, y0, z0), (x1, h10, z0), (x0, h00, z0)], col, (0, 0, -1))
            self.face([(x0, y0, z1), (x1, y0, z1), (x1, h11, z1), (x0, h01, z1)], col, (0, 0, 1))
            self.face([(x0, y0, z0), (x0, y0, z1), (x0, h01, z1), (x0, h00, z0)], col, (-1, 0, 0))
            self.face([(x1, y0, z0), (x1, y0, z1), (x1, h11, z1), (x1, h10, z0)], col, (1, 0, 0))
        if solid and not self.dry:
            p = self.P(x0, y0, z0)
            q = self.P(x1, max(a, b), z1)
            c = Surface(1)
            c.x0, c.x1 = min(p[0], q[0]), max(p[0], q[0])
            c.z0, c.z1 = min(p[2], q[2]), max(p[2], q[2])
            c.y0 = p[1]
            c.y1 = q[1]
            c.axis = axis
            c.a = self.oy + self.s * a
            c.b = self.oy + self.s * b
            if axis == 0 and self.mx < 0:
                c.a, c.b = c.b, c.a
            self.cols.append(c)

    def prism(self, cx, cz, y0, y1, r, sides, col, top=None, rt=None, solid=True, rot=0.0, bottom=False,
              cap=True, inward=False, raw=False, colr=None):
        if rt is None:
            rt = r
        if not self.dry:
            A = [rot + TAU * i / sides for i in range(sides)]
            B = [(cx + math.cos(a) * r, y0, cz + math.sin(a) * r) for a in A]
            T = [(cx + math.cos(a) * rt, y1, cz + math.sin(a) * rt) for a in A]
            sg = -1 if inward else 1
            for i in range(sides):
                j = (i + 1) % sides
                am = rot + TAU * (i + 0.5) / sides
                hint = (math.cos(am) * sg, 0.0, math.sin(am) * sg)
                if r <= 1e-6:
                    q = [B[i], T[j], T[i]]
                elif rt <= 1e-6:
                    q = [B[i], B[j], T[i]]
                else:
                    q = [B[i], B[j], T[j], T[i]]
                c = colr[i % len(colr)] if colr else col
                self.face(q, c, hint, raw=raw)
            if cap and rt > 1e-6:
                self.face(T, top if top is not None else col, (0, sg, 0), raw=raw)
            if bottom and r > 1e-6:
                self.face(B, col, (0, -sg, 0), raw=raw)
        if solid and not self.dry:
            p = self.P(cx, y0, cz)
            q = self.P(cx, y1, cz)
            c = Surface(4 if inward else 2)
            c.cx, c.cz = p[0], p[2]
            c.y0, c.y1 = min(p[1], q[1]), max(p[1], q[1])
            if inward:
                c.r = min(r, rt) * self.s * math.cos(math.pi / sides)
            else:
                c.r = (max(r, rt) if min(r, rt) > 0 else max(r, rt) * 0.55) * self.s
            c.x0, c.x1, c.z0, c.z1 = c.cx - c.r, c.cx + c.r, c.cz - c.r, c.cz + c.r
            self.cols.append(c)

    def heightfield(self, x0, z0, cell, nx, nz, hfn, cfn, solid=True):
        H = [[hfn(x0 + i * cell, z0 + j * cell) for i in range(nx + 1)] for j in range(nz + 1)]
        if not self.dry:
            for j in range(nz):
                for i in range(nx):
                    xa = x0 + i * cell
                    za = z0 + j * cell
                    p00 = (xa, H[j][i], za)
                    p10 = (xa + cell, H[j][i + 1], za)
                    p11 = (xa + cell, H[j + 1][i + 1], za + cell)
                    p01 = (xa, H[j + 1][i], za + cell)
                    hm = (p00[1] + p11[1]) * 0.5
                    c = cfn(xa + cell * 0.5, za + cell * 0.5, hm, i, j)
                    self.face([p00, p10, p11], c, (0, 1, 0))
                    self.face([p00, p11, p01], cmul(c, 0.94), (0, 1, 0))
            if solid:
                c = Surface(3)
                p = self.P(x0, 0, z0)
                c.x0, c.z0 = p[0], p[2]
                c.cell = cell * self.s
                c.nx, c.nz = nx, nz
                c.x1, c.z1 = c.x0 + nx * c.cell, c.z0 + nz * c.cell
                c.H = [[self.oy + self.s * h for h in row] for row in H]
                c.y0 = -1e6
                c.y1 = max(max(r) for r in c.H)
                self.cols.append(c)
        return H

    def water_plane(self, y, x0, z0, x1, z1, col=None, tile=10.0):
        col = col or self.water_col
        self.water = self.P(0, y, 0)[1]
        self.water_col = col
        nx = max(1, int((x1 - x0) / tile))
        nz = max(1, int((z1 - z0) / tile))
        self.qgrid((x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1), nx, nz, col, (0, 1, 0),
                   col2=cmul(col, 1.08), dbl=True, raw=True, bias=0.3)

    def lava_plane(self, y, x0, z0, x1, z1, tile=10.0):
        self.lava = self.P(0, y, 0)[1]
        nx = max(1, int((x1 - x0) / tile))
        nz = max(1, int((z1 - z0) / tile))
        self.qgrid((x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1), nx, nz, (255, 90, 20), (0, 1, 0),
                   col2=(255, 150, 30), raw=True)

    def obox(self, cx, cy, cz, yaw, l0, l1, y0, y1, d0, d1, col, solid=True, **kw):
        """Oriented (yaw multiple of 90deg) box in portal-local lateral/depth coordinates."""
        fx, fz = round(math.sin(yaw)), round(math.cos(yaw))
        rx, rz = fz, -fx
        xa = cx + rx * l0 + fx * d0
        za = cz + rz * l0 + fz * d0
        xb = cx + rx * l1 + fx * d1
        zb = cz + rz * l1 + fz * d1
        return self.box(xa, cy + y0, za, xb, cy + y1, zb, col, solid=solid, **kw)

    # -- game objects -------------------------------------------------------------
    def spawn(self, name, x, y, z, yaw=0.0):
        p = self.P(x, y, z)
        self.spawns[name] = (p[0], p[1], p[2], self.Y(yaw))

    def portal(self, name, x, y, z, yaw, kind='door', rules=None, w=None, h=None, lock=None, visible=None,
               fy=None, color=None, free=False):
        if self.suppress:
            return None
        if w is None:
            w = {'door': 2.2, 'painting': 3.2, 'hole': 2.4, 'pipe': 2.2, 'ring': 2.6}[kind]
        if h is None:
            h = {'door': 3.4, 'painting': 2.6, 'hole': 0.1, 'pipe': 1.2, 'ring': 2.6}[kind]
        if rules is None:
            rules = []
        elif isinstance(rules, tuple) and rules and isinstance(rules[0], str):
            rules = [(None, rules)]
        rules = list(rules)
        rh = self.w.rare_hosts.get((self.aid, self.v, name))
        if rh:
            rid, k = rh
            pid = "%s:%d:%s" % (self.aid, self.v, name)
            rules.insert(0, ((lambda wd, pid=pid, k=k: wd.portal_uses[pid] == k - 1), (rid, 0, 'exit')))
        fcol = color or (200, 170, 90)
        if kind == 'door':
            if not self.dry:
                t = 0.28
                self.obox(x, y, z, yaw, -w / 2 - t, -w / 2, 0, h + t, -0.45, 0.35, fcol, solid=free)
                self.obox(x, y, z, yaw, w / 2, w / 2 + t, 0, h + t, -0.45, 0.35, fcol, solid=free)
                self.obox(x, y, z, yaw, -w / 2 - t, w / 2 + t, h, h + t, -0.45, 0.35, fcol, solid=free)
                if free:
                    self.obox(x, y, z, yaw, -w / 2, w / 2, 0, h, -0.45, -0.12, cmul(fcol, 0.7))
        elif kind == 'painting':
            self.obox(x, y, z, yaw, -w / 2 - 0.25, w / 2 + 0.25, -0.25, h + 0.25, -0.05, 0.08, fcol, solid=False,
                      bias=0.4)
        elif kind == 'hole':
            hw = w / 2
            for (a0, a1, b0, b1) in ((-hw - .3, hw + .3, -hw - .3, -hw), (-hw - .3, hw + .3, hw, hw + .3),
                                     (-hw - .3, -hw, -hw, hw), (hw, hw + .3, -hw, hw)):
                self.box(x + a0, y, z + b0, x + a1, y + 0.12, z + b1, fcol, solid=False, bias=0.3)
        elif kind == 'pipe':
            self.prism(x, z, y - h, y, w * 0.5, 10, (40, 170, 70), top=(20, 60, 30))
            self.prism(x, z, y - 0.35, y + 0.02, w * 0.5 + 0.18, 10, (60, 200, 90), solid=False, cap=False)
        elif kind == 'ring':
            if not self.dry:
                fx, fz = math.sin(yaw), math.cos(yaw)
                rx, rz = fz, -fx
                R = w * 0.5 + 0.1
                for i in range(12):
                    a = TAU * i / 12
                    px = x + rx * math.cos(a) * R
                    pz = z + rz * math.cos(a) * R
                    py = y + math.sin(a) * R
                    self.box(px - .22, py - .22, pz - .22, px + .22, py + .22, pz + .22,
                             hsv(i / 12.0, 0.5, 1.0), solid=False, raw=True)
        P = self.P(x, y, z)
        fyv = self.P(0, fy if fy is not None else (y - h * 0.5 if kind == 'ring' else y), 0)[1]
        p = WarpNode(self.aid, self.v, name, kind, P[0], P[1], P[2], self.Y(yaw), w * self.s, h * self.s,
                   rules, lock, visible, fyv, fcol)
        self.portals.append(p)
        if name not in self.spawns:
            ex = p.exit_point()
            self.spawns[name] = (ex[0], ex[1], ex[2], p.yaw)
        return p

    def star(self, x, y, z, cond=None, final=False, label=None):
        if self.suppress:
            return
        sid = "%s:%d:%d" % (self.aid, self.v, self.nstar)
        self.nstar += 1
        p = self.P(x, y, z)
        self.power_stars.append(StarDef(sid, p[0], p[1], p[2], cond, final, label))

    def coin(self, x, y, z):
        if not self.suppress:
            self.coins.append(self.P(x, y, z))

    def coins_line(self, a, b, n):
        for i in range(n):
            t = i / max(1, n - 1)
            self.coin(lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t))

    def coins_ring(self, cx, cy, cz, r, n):
        for i in range(n):
            a = TAU * i / n
            self.coin(cx + math.cos(a) * r, cy, cz + math.sin(a) * r)

    def npc(self, x, y, z, name, lines, style=0, yaw=0.0, col=None):
        if not self.suppress:
            p = self.P(x, y, z)
            self.npcs.append((p, name, lines, style, self.Y(yaw), col))

    def enemy(self, kind, x, y, z, rad=5.0, **kw):
        if not self.suppress:
            p = self.P(x, y, z)
            self.enemies.append((kind, p, rad * self.s, kw))

    def switch(self, x, y, z, flag, msg="SOMETHING CHANGED SOMEWHERE."):
        if not self.suppress:
            self.switches.append((self.P(x, y, z), flag, msg))

    def trigger(self, x0, y0, z0, x1, y1, z1, kind, data, msg=None):
        if self.suppress:
            return
        a = self.P(x0, y0, z0)
        b = self.P(x1, y1, z1)
        self.triggers.append([min(a[0], b[0]), min(a[1], b[1]), min(a[2], b[2]),
                              max(a[0], b[0]), max(a[1], b[1]), max(a[2], b[2]), kind, data, msg])

    def fake_wall(self, x0, y0, z0, x1, y1, z1, col, name, **kw):
        self.box(x0, y0, z0, x1, y1, z1, col, solid=False, **kw)
        self.trigger(x0, y0, z0, x1, y1, z1, 'secret', name, "FAKE WALL!  SECRET FOUND")

    def mover(self, x, y, z, hx, hy, hz, kind, amp, speed, phase=0.0, color=(200, 200, 210)):
        if self.dry:
            return
        c = Surface(0)
        p = self.P(x, y, z)
        s = self.s
        a = (amp[0] * s * (self.mx if kind != 'circle' else 1), amp[1] * s, amp[2] * s)
        m = MovingPlatform(c, hx * s, hy * s, hz * s, p, kind, a, speed, phase, color)
        self.movers.append(m)

    # -- finalise -------------------------------------------------------------------
    def finalize(self, info):
        a = Area()
        for k in ('portals', 'npcs', 'enemies', 'switches', 'triggers', 'spawns', 'movers', 'cols'):
            setattr(a, k, getattr(self, k))
        a.stars = self.power_stars
        a.coins = self.coins
        for k in ('sky', 'fog', 'water', 'water_col', 'lava', 'kill_y', 'starfield', 'wobble', 'hue_cycle', 'view',
                  'fog_near', 'glitch', 'amb'):
            setattr(a, k, getattr(self, k))
        a.aid, a.v = self.aid, self.v
        a.name, a.music = info
        if self.music:
            a.music = self.music
        a.seed = seed_hash(self.w.seed, self.aid, self.v)
        if self.dry:
            return a
        CH = 16.0
        buckets = {}
        for tp, c, n, dbl, bias in self.faces:
            L = len(tp)
            cx = sum(p[0] for p in tp) / L
            cy = sum(p[1] for p in tp) / L
            cz = sum(p[2] for p in tp) / L
            key = (int(math.floor(cx / CH)), int(math.floor(cy / CH)), int(math.floor(cz / CH)))
            buckets.setdefault(key, []).append((tp, c, n, dbl, bias, cx, cy, cz))
        for fl in buckets.values():
            m = GfxMesh()
            vmap = {}
            V = m.verts
            for tp, c, n, dbl, bias, cx, cy, cz in fl:
                idx = []
                for p in tp:
                    kk = (round(p[0], 3), round(p[1], 3), round(p[2], 3))
                    i = vmap.get(kk)
                    if i is None:
                        i = len(V)
                        vmap[kk] = i
                        V.append(p)
                    idx.append(i)
                d = n[0] * tp[0][0] + n[1] * tp[0][1] + n[2] * tp[0][2]
                m.faces.append((tuple(idx), n[0], n[1], n[2], d, c, dbl, cx, cy, cz, bias))
            xs = [p[0] for p in V]
            ys = [p[1] for p in V]
            zs = [p[2] for p in V]
            m.cx = (min(xs) + max(xs)) * 0.5
            m.cy = (min(ys) + max(ys)) * 0.5
            m.cz = (min(zs) + max(zs)) * 0.5
            m.rad = max(math.sqrt((p[0] - m.cx) ** 2 + (p[1] - m.cy) ** 2 + (p[2] - m.cz) ** 2) for p in V)
            a.meshes.append(m)
        a.nfaces = len(self.faces)
        a.index()
        return a


# ----------------------------------------------------------------------------
# /* src/game/camera.c */
# ----------------------------------------------------------------------------
class Camera:  # /* src/game/camera.h */
    def __init__(self):
        self.yaw = 0.0
        self.pitch = 0.32
        self.dist = 7.5
        self.cur = 7.5
        self.pos = Vec3f(0, 3, -8)
        self.target = Vec3f(0, 1, 0)
        self.fwd = (0.0, 0.0, 1.0)
        self.right = (1.0, 0.0, 0.0)
        self.up = (0.0, 1.0, 0.0)

    def snap(self, px, py, pz, yaw):
        self.yaw = yaw
        self.target.set(px, py + 1.3, pz)
        self.cur = self.dist
        self._place(None)

    def _place(self, area):
        cp = math.cos(self.pitch)
        ox = -math.sin(self.yaw) * cp
        oy = math.sin(self.pitch)
        oz = -math.cos(self.yaw) * cp
        t = self.target
        d = self.dist
        if area is not None:
            hit = d
            n = 10
            for i in range(1, n + 1):
                s = 0.6 + (d - 0.6) * i / n
                if area.cam_solid(t.x + ox * s, t.y + oy * s, t.z + oz * s):
                    hit = max(0.8, 0.6 + (d - 0.6) * (i - 1) / n - 0.3)
                    break
            if hit < self.cur:
                self.cur = hit
            else:
                self.cur += (hit - self.cur) * 0.08
        else:
            self.cur = d
        self.pos.set(t.x + ox * self.cur, t.y + oy * self.cur, t.z + oz * self.cur)
        self.basis()

    def update(self, px, py, pz, area, dt):
        k = 1.0 - math.exp(-14.0 * dt)
        t = self.target
        t.x += (px - t.x) * k
        t.y += (py + 1.3 - t.y) * (1.0 - math.exp(-8.0 * dt))
        t.z += (pz - t.z) * k
        self.pitch = clamp(self.pitch, -0.35, 1.25)
        self._place(area)

    def basis(self):
        fx = self.target.x - self.pos.x
        fy = self.target.y - self.pos.y
        fz = self.target.z - self.pos.z
        l = math.sqrt(fx * fx + fy * fy + fz * fz) or 1.0
        fx, fy, fz = fx / l, fy / l, fz / l
        hl = math.hypot(fx, fz) or 1e-6
        rx, rz = fz / hl, -fx / hl
        self.fwd = (fx, fy, fz)
        self.right = (rx, 0.0, rz)
        self.up = (fy * rz, fz * rx - fx * rz, -fy * rx)


# ----------------------------------------------------------------------------
# /* src/engine/graph_node.c + rendering */  GraphRenderer
# ----------------------------------------------------------------------------
class GraphRenderer:  # /* src/engine/graph_node */
    def __init__(self):
        self.surf = pygame.Surface((RW, RH))
        self.near = 0.12
        self.f = (RH * 0.5) / math.tan(math.radians(36))
        self.tx = (RW * 0.5) / self.f
        self.ty = (RH * 0.5) / self.f
        self.kx = math.sqrt(1 + self.tx * self.tx)
        self.ky = math.sqrt(1 + self.ty * self.ty)
        self.dl = []
        self.sky_cache = {}
        self.star_cache = {}
        self.stats_faces = 0
        self.stats_chunks = 0
        self.jitter = True
        self.fog_on = True
        self.rd = 110.0

    # -- per-frame setup ------------------------------------------------------------
    def begin(self, cam, area, settings, t):
        self.cpos = (cam.pos.x, cam.pos.y, cam.pos.z)
        self.fwd = cam.fwd
        self.right = cam.right
        self.up = cam.up
        self.jitter = settings.jitter
        self.fog_on = settings.fog
        self.rd = min(settings.render_dist, area.view)
        self.fs = self.rd * area.fog_near
        self.finv = 1.0 / max(1e-3, self.rd - self.fs)
        fog = area.fog
        sky = area.sky
        if area.hue_cycle:
            sh = math.sin(t * 0.21) * 0.5 + 0.5
            fog = lerpc(fog, (fog[2], fog[0], fog[1]), sh)
            sky = (lerpc(sky[0], (sky[0][1], sky[0][2], sky[0][0]), sh), lerpc(sky[1], fog, 0.5), sky[2])
        self.fogc = fog
        self.dl = []
        self.stats_faces = 0
        self.stats_chunks = 0
        self.draw_sky(sky, fog, area, t)

    def draw_sky(self, sky, fog, area, t):
        key = (sky, fog)
        g = self.sky_cache.get(key)
        if g is None:
            if len(self.sky_cache) > 40:
                self.sky_cache.clear()
            H = RH * 3
            col = pygame.Surface((1, H))
            hz = H // 2
            top, hor, bot = sky
            for y in range(H):
                if y < hz:
                    tt = (y / hz) ** 1.6
                    c = lerpc(top, hor, tt)
                    if y > hz - 14:
                        c = lerpc(c, fog, (y - (hz - 14)) / 14.0)
                else:
                    tt = min(1.0, (y - hz) / (hz * 0.5))
                    c = lerpc(fog, bot, tt)
                col.set_at((0, y), c)
            g = pygame.transform.scale(col, (RW, H))
            self.sky_cache[key] = g
        fx, fy, fz = self.fwd
        hl = math.hypot(fx, fz) or 1e-6
        hx, hz_ = fx / hl, fz / hl
        up = self.up
        yc = hx * up[0] + hz_ * up[2]
        zc = hx * fx + hz_ * fz
        hy = RH * 0.5 - self.f * yc / max(zc, 0.05)
        oy = int(hy - RH * 1.5)
        oy = clamp(oy, -RH * 2, 0)
        self.surf.blit(g, (0, oy))
        if area.starfield:
            stars = self.star_cache.get(area.seed)
            if stars is None:
                rng = random.Random(area.seed)
                stars = []
                for i in range(170):
                    a = rng.random() * TAU
                    e = rng.uniform(0.04, 1.0)
                    ce = math.sqrt(1 - e * e)
                    stars.append((math.cos(a) * ce, e, math.sin(a) * ce, rng.choice((1, 1, 1, 2)),
                                  rng.randint(170, 255)))
                self.star_cache[area.seed] = stars
            rx, _, rz = self.right
            ux, uy, uz = self.up
            f = self.f
            s = self.surf
            for dx, dy, dz, sz, br in stars:
                zc = dx * fx + dy * fy + dz * fz
                if zc < 0.1:
                    continue
                sx = RW * 0.5 + (dx * rx + dz * rz) * f / zc
                sy = RH * 0.5 - (dx * ux + dy * uy + dz * uz) * f / zc
                if 0 <= sx < RW and 0 <= sy < RH:
                    b = int(br * (0.75 + 0.25 * math.sin(t * 3 + dx * 40)))
                    s.fill((b, b, min(255, b + 20)), (int(sx), int(sy), sz, sz))

    # -- clipping --------------------------------------------------------------------
    def _clip(self, cp):
        near = self.near
        out = []
        L = len(cp)
        for i in range(L):
            a = cp[i]
            b = cp[(i + 1) % L]
            ain = a[2] > near
            bin_ = b[2] > near
            if ain:
                out.append(a)
            if ain != bin_:
                t = (near - a[2]) / (b[2] - a[2])
                out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, near))
        if len(out) < 3:
            return None
        f = self.f
        hw = RW * 0.5
        hh = RH * 0.5
        return [(hw + x * f / z, hh - y * f / z) for x, y, z in out]

    @staticmethod
    def _clip2d(poly):
        """Clip a screen polygon to a guard band (huge coordinates make scanline fills slow)."""
        G = 300.0
        pts = list(poly)
        for axis, lim, keep_less in ((0, -G, False), (0, RW + G, True), (1, -G, False), (1, RH + G, True)):
            if not pts:
                return None
            out = []
            L = len(pts)
            for i in range(L):
                a = pts[i]
                b = pts[(i + 1) % L]
                ain = (a[axis] <= lim) if keep_less else (a[axis] >= lim)
                bin_ = (b[axis] <= lim) if keep_less else (b[axis] >= lim)
                if ain:
                    out.append(a)
                if ain != bin_:
                    t = (lim - a[axis]) / (b[axis] - a[axis])
                    if axis == 0:
                        out.append((lim, a[1] + (b[1] - a[1]) * t))
                    else:
                        out.append((a[0] + (b[0] - a[0]) * t, lim))
            pts = out
        return pts if len(pts) >= 3 else None

    # -- static world ----------------------------------------------------------------
    def draw_static(self, meshes, wob=0.0):
        px, py, pz = self.cpos
        fx, fy, fz = self.fwd
        rx, _, rz = self.right
        ux, uy, uz = self.up
        f = self.f
        hw = RW * 0.5
        hh = RH * 0.5
        near = self.near
        rd = self.rd
        tx, ty, kx, ky = self.tx, self.ty, self.kx, self.ky
        ap = self.dl.append
        fog = self.fog_on
        fs = self.fs
        finv = self.finv
        fr, fg, fb = self.fogc
        jit = self.jitter
        clip = self._clip
        c2d = self._clip2d
        nf = 0
        nc = 0
        for m in meshes:
            dx = m.cx - px
            dy = m.cy - py
            dz = m.cz - pz
            R = m.rad
            zc = dx * fx + dy * fy + dz * fz
            if zc < -R or zc - R > rd:
                continue
            xc = dx * rx + dz * rz
            if abs(xc) - zc * tx > R * kx:
                continue
            yc = dx * ux + dy * uy + dz * uz
            if abs(yc) - zc * ty > R * ky:
                continue
            nc += 1
            chk = zc - R < 40.0
            if jit:
                pts = [(int(((x - px) * rx + (z - pz) * rz) * 16) * 0.0625,
                        int(((x - px) * ux + (y - py) * uy + (z - pz) * uz) * 16) * 0.0625,
                        (x - px) * fx + (y - py) * fy + (z - pz) * fz) for x, y, z in m.verts]
            else:
                pts = [((x - px) * rx + (z - pz) * rz,
                        (x - px) * ux + (y - py) * uy + (z - pz) * uz,
                        (x - px) * fx + (y - py) * fy + (z - pz) * fz) for x, y, z in m.verts]
            if wob:
                pts = [(a + math.sin(c * 0.7 + wob) * 0.08 * c * 0.1, b + math.cos(a * 0.5 + wob) * 0.06 * c * 0.1, c)
                       for a, b, c in pts]
            proj = [(hw + a * f / c, hh - b * f / c) if c > near else None for a, b, c in pts]
            for idx, nx, ny, nz, d, col, dbl, cx, cy, cz, bias in m.faces:
                if nx * px + ny * py + nz * pz < d and not dbl:
                    continue
                z = (cx - px) * fx + (cy - py) * fy + (cz - pz) * fz
                if z > rd:
                    continue
                if len(idx) == 3:
                    a, b, c = idx
                    pa = proj[a]
                    pb = proj[b]
                    pc = proj[c]
                    if pa and pb and pc:
                        poly = (pa, pb, pc)
                    else:
                        poly = clip([pts[a], pts[b], pts[c]])
                        if poly is None:
                            continue
                elif len(idx) == 4:
                    a, b, c, e = idx
                    pa = proj[a]
                    pb = proj[b]
                    pc = proj[c]
                    pe = proj[e]
                    if pa and pb and pc and pe:
                        poly = (pa, pb, pc, pe)
                    else:
                        poly = clip([pts[a], pts[b], pts[c], pts[e]])
                        if poly is None:
                            continue
                else:
                    ps = [proj[i] for i in idx]
                    if None in ps:
                        poly = clip([pts[i] for i in idx])
                        if poly is None:
                            continue
                    else:
                        poly = ps
                if chk:
                    for q in poly:
                        if q[0] < -300 or q[0] > 726 or q[1] < -300 or q[1] > 540:
                            poly = c2d(poly)
                            break
                    if poly is None:
                        continue
                if fog and z > fs:
                    t = (z - fs) * finv
                    if t > 1.0:
                        t = 1.0
                    col = (int(col[0] + (fr - col[0]) * t), int(col[1] + (fg - col[1]) * t),
                           int(col[2] + (fb - col[2]) * t))
                ap((z - bias, col, poly))
                nf += 1
        self.stats_faces += nf
        self.stats_chunks = nc

    # -- dynamic geometry --------------------------------------------------------------
    def face(self, pts, col, n=None, bias=0.0):
        px, py, pz = self.cpos
        if n is not None:
            p = pts[0]
            if n[0] * (px - p[0]) + n[1] * (py - p[1]) + n[2] * (pz - p[2]) <= 0:
                return
        fx, fy, fz = self.fwd
        rx, _, rz = self.right
        ux, uy, uz = self.up
        cp = [((x - px) * rx + (z - pz) * rz, (x - px) * ux + (y - py) * uy + (z - pz) * uz,
               (x - px) * fx + (y - py) * fy + (z - pz) * fz) for x, y, z in pts]
        z = sum(c[2] for c in cp) / len(cp)
        if z > self.rd:
            return
        near = self.near
        if all(c[2] > near for c in cp):
            f = self.f
            hw = RW * 0.5
            hh = RH * 0.5
            poly = [(hw + a * f / c, hh - b * f / c) for a, b, c in cp]
        else:
            poly = self._clip(cp)
            if poly is None:
                return
        for q in poly:
            if q[0] < -300 or q[0] > 726 or q[1] < -300 or q[1] > 540:
                poly = self._clip2d(poly)
                if poly is None:
                    return
                break
        if self.fog_on and z > self.fs:
            t = min(1.0, (z - self.fs) * self.finv)
            col = lerpc(col, self.fogc, t)
        self.dl.append((z - bias, col, poly))
        self.stats_faces += 1

    def near_view(self, x, y, z, r=2.0):
        px, py, pz = self.cpos
        dx, dy, dz = x - px, y - py, z - pz
        fx, fy, fz = self.fwd
        zc = dx * fx + dy * fy + dz * fz
        if zc < -r or zc - r > self.rd:
            return False
        rx, _, rz = self.right
        return abs(dx * rx + dz * rz) - zc * self.tx <= r * self.kx

    def box(self, px, py, pz, hx, hy, hz, col, yaw=0.0, pitch=0.0, off=(0.0, 0.0, 0.0), bias=0.0, lit=True):
        cy, sy = math.cos(yaw), math.sin(yaw)
        cpt, spt = math.cos(pitch), math.sin(pitch)
        ox, oy, oz = off

        def xf(lx, ly, lz):
            y2 = ly * cpt - lz * spt
            z2 = ly * spt + lz * cpt
            return (px + lx * cy + z2 * sy, py + y2, pz - lx * sy + z2 * cy)

        def nf(lx, ly, lz):
            y2 = ly * cpt - lz * spt
            z2 = ly * spt + lz * cpt
            return (lx * cy + z2 * sy, y2, -lx * sy + z2 * cy)

        c = [xf(ox + sx * hx, oy + sy_ * hy, oz + sz * hz) for sx, sy_, sz in
             ((-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1))]
        for ids, n in (((0, 1, 2, 3), (0, 0, -1)), ((5, 4, 7, 6), (0, 0, 1)), ((4, 0, 3, 7), (-1, 0, 0)),
                       ((1, 5, 6, 2), (1, 0, 0)), ((3, 2, 6, 7), (0, 1, 0)), ((4, 5, 1, 0), (0, -1, 0))):
            wn = nf(*n)
            self.face([c[i] for i in ids], shade(col, wn) if lit else col, wn, bias)

    def octa(self, x, y, z, r, h, yaw, col, bias=0.3, lit=True):
        pts = [(x + math.cos(yaw + i * HALF_PI) * r, y, z + math.sin(yaw + i * HALF_PI) * r) for i in range(4)]
        top = (x, y + h, z)
        bot = (x, y - h, z)
        for i in range(4):
            a = pts[i]
            b = pts[(i + 1) % 4]
            for apex, sgn in ((top, 1), (bot, -1)):
                tri = [a, b, apex] if sgn > 0 else [b, a, apex]
                n = newell(tri) or (0, sgn, 0)
                mx = (a[0] + b[0]) * 0.5 - x
                mz = (a[2] + b[2]) * 0.5 - z
                if n[0] * mx + n[2] * mz < 0:
                    n = (-n[0], -n[1], -n[2])
                self.face(tri, shade(col, n) if lit else col, n, bias)

    def star(self, x, y, z, R, yaw, col, bias=0.4):
        """B3313 Power Star (Power Star mesh): thick five-pointed star, upright, spinning."""
        cy, sy = math.cos(yaw), math.sin(yaw)
        th = R * 0.32
        fc = (x + sy * th, y, z + cy * th)
        bc = (x - sy * th, y, z - cy * th)
        ring = []
        for i in range(10):
            a = HALF_PI + TAU * i / 10
            rr = R if i % 2 == 0 else R * 0.45
            lx = math.cos(a) * rr
            ly = math.sin(a) * rr
            ring.append((x + lx * cy, y + ly, z - lx * sy))
        for i in range(10):
            a = ring[i]
            b = ring[(i + 1) % 10]
            lum = 0.75 + 0.25 * math.sin(i * 1.3 + yaw)
            c1 = cmul(col, lum + 0.15)
            c2 = cmul(col, lum - 0.1)
            n1 = newell([a, b, fc])
            n2 = newell([b, a, bc])
            if n1:
                if (n1[0] * sy + n1[2] * cy) < 0:
                    n1 = (-n1[0], -n1[1], -n1[2])
                self.face([a, b, fc], c1, n1, bias)
            if n2:
                if (n2[0] * sy + n2[2] * cy) > 0:
                    n2 = (-n2[0], -n2[1], -n2[2])
                self.face([b, a, bc], c2, n2, bias)

    def flat_poly(self, pts, col, bias=0.5):
        self.face(pts, col, None, bias)

    def flush(self):
        dl = self.dl
        dl.sort(key=itemgetter(0), reverse=True)
        dp = pygame.draw.polygon
        s = self.surf
        for z, col, poly in dl:
            dp(s, col, poly)
        self.dl = []


# ----------------------------------------------------------------------------
# /* levels/level_helpers.c */
# ----------------------------------------------------------------------------
def has(f):
    return lambda w: f in w.flags


def need(k):
    return lambda w: len(w.stars) >= k


def visited(a):
    return lambda w: w.visits[a] > 0


def make_box(ab, x0, y0, z0, x1, y1, z1, col, **kw):
    return ab.box(x0, y0, z0, x1, y1, z1, col, **kw)


def make_platform(ab, x, y, z, w, d, col, h=0.8, **kw):
    return ab.box(x - w / 2, y - h, z - d / 2, x + w / 2, y, z + d / 2, col, **kw)


def make_ramp(ab, x0, z0, x1, z1, y0, a, b, axis, col, **kw):
    return ab.ramp(x0, z0, x1, z1, y0, a, b, axis, col, **kw)


def make_portal(ab, *a, **kw):
    return ab.portal(*a, **kw)


def wall_x(ab, x0, x1, z, y0, y1, col, gaps=(), t=0.6, **kw):
    cur = x0
    for g in sorted(gaps, key=lambda g: g[0]):
        c, w, h = g[0], g[1], g[2]
        a, b = c - w / 2, c + w / 2
        if a > cur:
            ab.box(cur, y0, z - t / 2, a, y1, z + t / 2, col, **kw)
        if y0 + h < y1:
            ab.box(a, y0 + h, z - t / 2, b, y1, z + t / 2, col, **kw)
        if len(g) > 3:
            ab.fake_wall(a, y0, z - t / 2, b, y0 + h, z + t / 2, col, g[4])
        cur = b
    if cur < x1:
        ab.box(cur, y0, z - t / 2, x1, y1, z + t / 2, col, **kw)


def wall_z(ab, z0, z1, x, y0, y1, col, gaps=(), t=0.6, **kw):
    cur = z0
    for g in sorted(gaps, key=lambda g: g[0]):
        c, w, h = g[0], g[1], g[2]
        a, b = c - w / 2, c + w / 2
        if a > cur:
            ab.box(x - t / 2, y0, cur, x + t / 2, y1, a, col, **kw)
        if y0 + h < y1:
            ab.box(x - t / 2, y0 + h, a, x + t / 2, y1, b, col, **kw)
        if len(g) > 3:
            ab.fake_wall(x - t / 2, y0, a, x + t / 2, y0 + h, b, col, g[4])
        cur = b
    if cur < z1:
        ab.box(x - t / 2, y0, cur, x + t / 2, y1, z1, col, **kw)


def make_room(ab, x0, z0, x1, z1, y, h, wall, floor, ceil=None, gaps=None, checker=None, t=0.6, tile=4.0):
    gaps = gaps or {}
    if floor is not None:
        ab.box(x0 - t / 2, y - 1, z0 - t / 2, x1 + t / 2, y, z1 + t / 2, floor, checker=checker, tile=tile)
    if gaps.get('s', ()) is not None:
        wall_x(ab, x0 - t / 2, x1 + t / 2, z0, y, y + h, wall, gaps.get('s', ()), t)
    if gaps.get('n', ()) is not None:
        wall_x(ab, x0 - t / 2, x1 + t / 2, z1, y, y + h, wall, gaps.get('n', ()), t)
    if gaps.get('w', ()) is not None:
        wall_z(ab, z0 + t / 2, z1 - t / 2, x0, y, y + h, wall, gaps.get('w', ()), t)
    if gaps.get('e', ()) is not None:
        wall_z(ab, z0 + t / 2, z1 - t / 2, x1, y, y + h, wall, gaps.get('e', ()), t)
    if ceil is not None:
        ab.box(x0 - t / 2, y + h, z0 - t / 2, x1 + t / 2, y + h + 0.5, z1 + t / 2, ceil, tile=8.0)


def make_corridor(ab, x0, z0, x1, z1, y, h, wall, floor, ceil=None, t=0.5):
    ab.box(x0, y - 1, z0, x1, y, z1, floor)
    if (x1 - x0) >= (z1 - z0):
        wall_x(ab, x0, x1, z0, y, y + h, wall, (), t)
        wall_x(ab, x0, x1, z1, y, y + h, wall, (), t)
    else:
        wall_z(ab, z0, z1, x0, y, y + h, wall, (), t)
        wall_z(ab, z0, z1, x1, y, y + h, wall, (), t)
    if ceil is not None:
        ab.box(x0, y + h, z0, x1, y + h + 0.4, z1, ceil)


def make_stairs(ab, x, z, y0, n, dx, dz, width, rise, run, col, col2=None):
    for i in range(n):
        a0, a1 = i * run, (i + 1) * run
        top = y0 + rise * (i + 1)
        c = col2 if (col2 and i % 2) else col
        if dx:
            xa, xb = x + dx * a0, x + dx * a1
            ab.box(min(xa, xb), y0, z - width / 2, max(xa, xb), top, z + width / 2, c)
        else:
            za, zb = z + dz * a0, z + dz * a1
            ab.box(x - width / 2, y0, min(za, zb), x + width / 2, top, max(za, zb), c)


def make_tower(ab, x, z, y0, h, r, col, roof=None, sides=8, crenel=True, top=None):
    ab.prism(x, z, y0, y0 + h, r, sides, col, top=top)
    if roof is not None:
        ab.prism(x, z, y0 + h, y0 + h + r * 1.7, r + 0.5, sides, roof, rt=0.0, solid=False)
    elif crenel:
        for i in range(sides):
            a = TAU * i / sides
            cx, cz = x + math.cos(a) * (r - 0.35), z + math.sin(a) * (r - 0.35)
            ab.box(cx - .35, y0 + h, cz - .35, cx + .35, y0 + h + 0.7, cz + .35, col)


def make_arch(ab, x, z, y0, w, h, d, axis, col):
    t = 0.9
    if axis == 0:
        ab.box(x - w / 2 - t, y0, z - d / 2, x - w / 2, y0 + h, z + d / 2, col)
        ab.box(x + w / 2, y0, z - d / 2, x + w / 2 + t, y0 + h, z + d / 2, col)
        ab.box(x - w / 2 - t, y0 + h, z - d / 2, x + w / 2 + t, y0 + h + t, z + d / 2, col)
    else:
        ab.box(x - d / 2, y0, z - w / 2 - t, x + d / 2, y0 + h, z - w / 2, col)
        ab.box(x - d / 2, y0, z + w / 2, x + d / 2, y0 + h, z + w / 2 + t, col)
        ab.box(x - d / 2, y0 + h, z - w / 2 - t, x + d / 2, y0 + h + t, z + w / 2 + t, col)


def make_tree(ab, x, z, y0, s=1.0, trunk=(110, 75, 45), leaf=(40, 140, 60), kind='round'):
    ab.prism(x, z, y0 - 0.3, y0 + 2.2 * s, 0.32 * s, 5, trunk, solid=True)
    if kind == 'pine':
        for i in range(3):
            yb = y0 + (1.2 + i * 1.3) * s
            ab.prism(x, z, yb, yb + 2.0 * s, (1.8 - i * 0.45) * s, 6, cmul(leaf, 1 + i * 0.07), rt=0.0,
                     solid=False, bottom=True, rot=i * 0.4)
    elif kind == 'palm':
        ab.prism(x, z, y0 + 2.2 * s, y0 + 5.5 * s, 0.26 * s, 5, trunk, solid=True)
        ab.prism(x, z, y0 + 5.3 * s, y0 + 6.0 * s, 2.4 * s, 6, leaf, rt=0.4 * s, solid=False, bottom=True)
    else:
        ab.prism(x, z, y0 + 1.6 * s, y0 + 2.7 * s, 0.7 * s, 6, leaf, rt=1.8 * s, solid=False, cap=False)
        ab.prism(x, z, y0 + 2.7 * s, y0 + 4.0 * s, 1.8 * s, 6, cmul(leaf, 1.1), rt=0.35 * s, solid=False)


def make_island(ab, x, y, z, r, top=(90, 190, 90), rock=(150, 120, 100), sides=7, rot=0.0):
    ab.prism(x, z, y - 0.9, y, r, sides, rock, top=top, rot=rot)
    ab.prism(x, z, y - 0.9 - r * 1.25, y - 0.9, 0.0, sides, cmul(rock, 0.8), rt=r, solid=False, cap=False, rot=rot)


def make_mountain(ab, x, z, y0, r, h, layers, colfn, sides=9):
    for i in range(layers):
        rr = r * (1 - i / layers * 0.85)
        yb = y0 + h * i / layers
        yt = y0 + h * (i + 1) / layers
        ab.prism(x, z, yb, yt, rr, sides, colfn(i), rot=i * 0.37)


def spiral_steps(ab, cx, cz, y0, n, r, da, rise, a0, col, size=0.8, step_fn=None):
    last = None
    for i in range(n):
        a = a0 + i * da
        px, pz = cx + math.cos(a) * r, cz + math.sin(a) * r
        top = y0 + rise * (i + 1)
        c = step_fn(i) if step_fn else col
        ab.box(px - size, top - 0.5, pz - size, px + size, top, pz + size, c)
        last = (px, top, pz, a)
    return last


def gen_maze(rng, N, extra=0.1):
    """Return (hw, vw): hw[i][j] wall between (i,j)-(i,j+1); vw[i][j] wall between (i,j)-(i+1,j)."""
    hw = [[True] * (N - 1) for _ in range(N)]
    vw = [[True] * N for _ in range(N - 1)]
    seen = [[False] * N for _ in range(N)]
    stack = [(N // 2, 0)]
    seen[N // 2][0] = True
    while stack:
        i, j = stack[-1]
        nb = []
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < N and 0 <= b < N and not seen[a][b]:
                nb.append((a, b, di, dj))
        if not nb:
            stack.pop()
            continue
        a, b, di, dj = rng.choice(nb)
        if di == 1:
            vw[i][j] = False
        elif di == -1:
            vw[a][j] = False
        elif dj == 1:
            hw[i][j] = False
        else:
            hw[i][b] = False
        seen[a][b] = True
        stack.append((a, b))
    for _ in range(int(N * N * extra)):
        if rng.random() < 0.5:
            hw[rng.randrange(N)][rng.randrange(N - 1)] = False
        else:
            vw[rng.randrange(N - 1)][rng.randrange(N)] = False
    return hw, vw


def maze_dist(N, hw, vw, start):
    dist = {start: 0}
    q = [start]
    for (i, j) in q:
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if not (0 <= a < N and 0 <= b < N) or (a, b) in dist:
                continue
            if di == 1 and vw[i][j]:
                continue
            if di == -1 and vw[a][j]:
                continue
            if dj == 1 and hw[i][j]:
                continue
            if dj == -1 and hw[i][b]:
                continue
            dist[(a, b)] = dist[(i, j)] + 1
            q.append((a, b))
    return dist


def build_grid_maze(ab, N, C, H, wall, floor, doors, extra=0.1, ceil=None, t=0.8, top_deco=None):
    """doors: {(side, index): name}.  Returns (cell_center_fn, dist_map, door_positions)."""
    ox = -N * C / 2
    hw, vw = gen_maze(ab.rng, N, extra)
    ab.box(ox, -1, ox, -ox, 0, -ox, floor, tile=C, checker=cmul(floor, 1.1))
    for i in range(N):
        for j in range(N - 1):
            if hw[i][j]:
                z = ox + (j + 1) * C
                ab.box(ox + i * C - t / 2, 0, z - t / 2, ox + (i + 1) * C + t / 2, H, z + t / 2, wall)
    for i in range(N - 1):
        for j in range(N):
            if vw[i][j]:
                x = ox + (i + 1) * C
                ab.box(x - t / 2, 0, ox + j * C - t / 2, x + t / 2, H, ox + (j + 1) * C + t / 2, wall)
    gs = {'s': [], 'n': [], 'w': [], 'e': []}
    pos = {}
    for (side, k), name in doors.items():
        c = ox + (k + 0.5) * C
        gs[side].append((c, 2.2, 3.4))
        if side == 's':
            pos[name] = (c, ox, 0.0)
        elif side == 'n':
            pos[name] = (c, -ox, math.pi)
        elif side == 'w':
            pos[name] = (ox, c, HALF_PI)
        else:
            pos[name] = (-ox, c, -HALF_PI)
    wall_x(ab, ox - t / 2, -ox + t / 2, ox, 0, H, wall, gs['s'], t)
    wall_x(ab, ox - t / 2, -ox + t / 2, -ox, 0, H, wall, gs['n'], t)
    wall_z(ab, ox, -ox, ox, 0, H, wall, gs['w'], t)
    wall_z(ab, ox, -ox, -ox, 0, H, wall, gs['e'], t)
    if ceil is not None:
        ab.box(ox - 1, H, ox - 1, -ox + 1, H + 0.5, -ox + 1, ceil, tile=C)
    cc = lambda i, j: (ox + (i + 0.5) * C, ox + (j + 0.5) * C)
    start = (N // 2, 0)
    dist = maze_dist(N, hw, vw, start)
    return cc, dist, pos


# ----------------------------------------------------------------------------
# /* levels/b3313/*/ */  AREA BUILDERS (B3313 1.0)
# ----------------------------------------------------------------------------
def grounds_h(seed, x, z):
    ns = (seed % 997) * 0.37
    n = fbm(x * 0.028 + ns, z * 0.028 - ns, seed, 3)
    h = (n - 0.45) * 10.0
    dd = max(abs(x) - 24, -40 - z, z - 62, 0.0)
    h *= smooth(dd / 14.0)
    d = math.hypot(x + 38, z + 8)
    if d < 13:
        h = lerp(h, -3.2, 1 - smooth((d - 7) / 6))
    r = max(abs(x), abs(z - 10))
    if r > 58:
        h += (r - 58) * 0.55
    return h


def build_grounds(ab, replica=False):
    w = ab.w
    seed = w.seed
    dusk = ab.v == 1 and not replica
    rng = ab.rng
    if not replica:
        if dusk:
            ab.sky, ab.fog, ab.amb = ((25, 15, 60), (235, 130, 95), (50, 30, 50)), (205, 120, 110), 0.5
        else:
            ab.sky, ab.fog = ((50, 100, 225), (185, 215, 255), (110, 140, 90)), (190, 212, 245)
    H = lambda x, z: grounds_h(seed, x, z)
    g1 = (70, 150, 60) if not dusk else (60, 95, 70)
    g2 = (95, 175, 70) if not dusk else (80, 110, 80)

    def cfn(x, z, h, i, j):
        if h < -1.2:
            return (150, 130, 90)
        if h > 7:
            return (130, 125, 120)
        k = vnoise(x * 0.2, z * 0.2, seed + 5)
        return lerpc(g1, g2, k) if (i + j) % 2 else lerpc(g2, g1, k * 0.7)

    ab.heightfield(-75, -65, 5, 30, 30, H, cfn)
    for (a, b, c, d) in ((-76, -66, -75, 86), (75, -66, 76, 86), (-76, -66, 76, -65), (-76, 85, 76, 86)):
        ab.box(a, -10, b, c, 60, d, (0, 0, 0), visible=False)
    ab.water_plane(-1.0, -52, -22, -24, 6, (50, 110, 200) if not dusk else (60, 60, 130))
    stone = (185, 180, 165) if not dusk else (140, 120, 125)
    ab.box(-2.5, -0.5, -42, 2.5, 0.06, 39, stone, tile=5)
    # castle
    wallc = (225, 215, 190) if not dusk else (170, 140, 150)
    roofc = (60, 90, 170) if not dusk else (90, 40, 70)
    ab.box(-18, 0, 40, 18, 14, 60, wallc, top=(200, 190, 170))
    ab.prism(0, 52, 14, 24, 7, 8, wallc)
    ab.prism(0, 52, 24, 31, 8, 8, roofc, rt=0.0, solid=False)
    for tx, tz in ((-18, 40), (18, 40), (-18, 60), (18, 60)):
        make_tower(ab, tx, tz, 0, 18, 3, wallc, roof=roofc)
    for x in (-12, -6, 6, 12):
        ab.box(x - 1, 7, 39.8, x + 1, 10, 40, (40, 40, 70) if not dusk else (255, 220, 120), solid=False,
               raw=dusk, bias=0.5)
    front = ('dark', 0, 'front') if dusk else ('castle', 0, 'front')
    ab.portal('front', 0, 0, 39.55, math.pi, 'door', front, w=3.0, h=4.4, free=True, color=(160, 110, 60))
    # spiral tower
    tg = H(42, -22)
    make_tower(ab, 42, -22, tg - 1, 13, 3.5, (190, 180, 170))
    ttop = tg + 12
    spiral_steps(ab, 42, -22, tg, 11, 5.3, 0.62, 1.0, 0.0, (160, 150, 140))
    ab.star(42, ttop + 1.0, -22)
    ab.portal('tower', 42, ttop + 3.6, -22, 0.0, 'ring', ('sky', 0, 'down'), fy=ttop)
    # big tree
    g2y = H(-44, 30)
    ab.prism(-44, 30, g2y - 1, g2y + 9, 2.2, 7, (120, 85, 50))
    ab.prism(-44, 30, g2y + 8.6, g2y + 9.4, 6.5, 8, (60, 150, 70), bottom=True)
    ab.prism(-44, 30, g2y + 11.4, g2y + 12.2, 4.0, 8, (70, 170, 80), bottom=True)
    for i, (rr, hh) in enumerate(((3.4, 1.4), (4.6, 3.0), (5.8, 4.6), (7.4, 6.2), (7.8, 7.8))):
        a = 0.3 + i * 0.95
        px, pz = -44 + math.cos(a) * rr, 30 + math.sin(a) * rr
        ab.box(px - .8, g2y + hh - .4, pz - .8, px + .8, g2y + hh, pz + .8, (110, 80, 50))
    if dusk:
        ab.portal('tree', -41.7, g2y, 30, HALF_PI, 'door', ('giant', 0, 'entry'), w=1.8, h=3.0, free=True)
    else:
        ab.portal('tree', -41.7, g2y, 30, HALF_PI, 'door', ('tiny', 0, 'entry'), w=1.8, h=3.0, free=True)
        ab.star(-44, g2y + 13.0, 30)
    # pipe to caverns
    pg = H(22, -34)
    ab.portal('pipe', 22, pg + 1.2, -34, 0.0, 'pipe', ('caverns', 0, 'pipe'))
    # pond switch + hidden hollow ring behind castle
    ab.switch(-38, -3.2, -8, 'grounds_sw', "A RING HUMS BEHIND THE CASTLE.")
    hy = H(0, 70)
    hollow_vis = None if dusk else has('grounds_sw')
    ab.portal('hollow', 0, hy + 2.2, 70, 0.0, 'ring', ('garden', 1 if dusk else 0, 'entry'), visible=hollow_vis,
              fy=hy)
    if not dusk:
        ab.star(8, H(8, 70) + 1.0, 70, cond=has('grounds_sw'))
    # trees & rocks
    for i in range(26):
        x = rng.uniform(-68, 68)
        z = rng.uniform(-60, 80)
        if abs(x) < 26 and -42 < z < 64:
            continue
        if math.hypot(x + 38, z + 8) < 14 or math.hypot(x - 42, z + 22) < 9 or math.hypot(x + 44, z - 30) < 10:
            continue
        make_tree(ab, x, z, H(x, z), rng.uniform(0.9, 1.5), leaf=(40, 140, 60) if not dusk else (50, 80, 70),
                  kind=rng.choice(('round', 'pine')))
    for i in range(6):
        x = rng.uniform(-20, 20)
        z = rng.uniform(-38, 30)
        if abs(x) < 4:
            continue
        ab.box(x - 1, 0, z - 1, x + 1, 0.9, z + 1, (150, 145, 140))
    if dusk:
        ab.ramp(18, 48, 46, 53, 0, 14.0, 0.0, 0, (150, 120, 120))
        ab.star(10, 15.0, 45)
        wg = H(-14, -12)
        ab.prism(-14, -12, wg - 0.5, wg + 1.0, 1.8, 10, (120, 110, 120), top=(20, 10, 30))
        ab.portal('well', -14, wg + 1.0, -12, 0.0, 'hole', ('city', 0, 'well'), w=2.0)
    if replica:
        return
    ab.spawn('entry', 0, 0.1, -36, 0.0)
    ab.npc(6, 0, -28, "TOAD", ["OH! A VISITOR. THE CASTLE HAS BEEN REARRANGING ITSELF AGAIN.",
                                   "COLLECT THE POWER STARS. THE DOORS LISTEN TO HOW MANY YOU CARRY.",
                                   "THE POND IS DEEPER THAN IT LOOKS. SOMETHING CLICKS DOWN THERE."], style=1,
           yaw=math.pi)
    for i, (x, z) in enumerate(((14, 5), (-12, 18), (30, -40), (-30, 45))):
        ab.enemy('blob', x, H(x, z) + 1, z, 6)
    ab.coins_ring(0, 0.8, -20, 5, 8)
    ab.coins_line((0, 0.8, 0), (0, 0.8, 30), 6)
    ab.coins_line((-38, -2.2, -2), (-38, -2.2, -14), 4)


CASTLE_STYLES = {
    'main': dict(wall=(222, 208, 176), floor=(176, 52, 52), floor2=(232, 220, 196), ceil=(150, 110, 90),
                 fog=(210, 196, 170), frame=(210, 170, 80), pillar=(205, 195, 165), carpet=(150, 20, 40), amb=0.62),
    'alt': dict(wall=(170, 210, 180), floor=(90, 60, 150), floor2=(210, 200, 230), ceil=(110, 150, 120),
                fog=(185, 165, 215), frame=(120, 200, 190), pillar=(160, 190, 200), carpet=(40, 120, 110), amb=0.62),
    'dark': dict(wall=(80, 45, 55), floor=(22, 20, 26), floor2=(120, 22, 34), ceil=(40, 20, 25),
                 fog=(45, 6, 16), frame=(150, 40, 40), pillar=(90, 60, 70), carpet=(90, 0, 10), amb=0.45),
    'mirror': dict(wall=(180, 225, 240), floor=(40, 170, 190), floor2=(235, 245, 250), ceil=(120, 180, 200),
                   fog=(200, 235, 245), frame=(230, 140, 60), pillar=(200, 230, 240), carpet=(220, 120, 30), amb=0.66),
}


def build_castle(ab, style):
    S = CASTLE_STYLES[style]
    if style == 'mirror':
        ab.mx = -1
    ab.amb = S['amb']
    ab.fog = S['fog']
    ab.sky = (cmul(S['fog'], 0.6), S['fog'], cmul(S['fog'], 0.5))
    ab.view = 150
    wall, fc = S['wall'], S['frame']
    make_room(ab, -20, -16, 20, 24, 0, 14, wall, S['floor'], S['ceil'], checker=S['floor2'],
              gaps={'s': [(0, 3.0, 4.4)], 'w': [(10, 2.2, 3.4), (19, 3.0, 4.0, 'fake', 'castle_closet')],
                    'e': [(10, 2.2, 3.4)], 'n': None})
    wall_x(ab, -20.3, 20.3, 24, 0, 6, wall, [(0, 2.2, 3.4)])
    wall_x(ab, -20.3, 20.3, 24, 6, 14, wall, [(-12, 2.2, 3.4), (0, 3.0, 4.4), (12, 2.2, 3.4)])
    ab.box(-19.7, 5.4, 16, 19.7, 6, 23.7, cmul(S['floor2'], 0.9), top=S['floor2'])
    make_stairs(ab, 0, 8, 0, 15, 0, 1, 8, 0.4, 8 / 15, S['carpet'], cmul(S['carpet'], 1.2))
    ab.box(-2, 0, -16, 2, 0.05, 8, S['carpet'], solid=False, bias=1.5, bottom=False)
    # closet
    ab.box(-27, -1, 15, -20, 0, 23, S['floor'])
    wall_x(ab, -27.3, -20, 15, 0, 5, wall)
    wall_x(ab, -27.3, -20, 23, 0, 5, wall)
    wall_z(ab, 15, 23, -27, 0, 5, wall)
    ab.box(-27.3, 5, 14.7, -20, 5.5, 23.3, S['ceil'])
    # pillars, chandelier
    for sx in (-10, 10):
        ab.prism(sx, 12, 0, 6.5, 1.0, 8, S['pillar'])
        ab.prism(sx, -6, 0, 14, 1.2, 8, S['pillar'])
    ab.prism(0, 11.5, 7.5, 8.0, 1.6, 8, (230, 200, 90), bottom=True)
    ab.prism(0, 11.5, 8.0, 14, 0.08, 4, (90, 80, 60), solid=False)
    for i in range(6):
        a = TAU * i / 6
        ab.box(math.cos(a) * 1.4 - .1, 8.0, 11.5 + math.sin(a) * 1.4 - .1, math.cos(a) * 1.4 + .1, 8.5,
               11.5 + math.sin(a) * 1.4 + .1, (255, 240, 170), solid=False, raw=True)
    if style == 'alt':
        rng = ab.rng
        for i in range(7):
            x, y, z = rng.uniform(-16, 16), rng.uniform(3, 11), rng.uniform(-12, 4)
            ab.box(x - .7, y - .7, z - .7, x + .7, y + .7, z + .7, hsv(rng.random(), .5, .9), solid=False)
        ab.hue_cycle = True
    if style == 'dark':
        ab.starfield = False
    # rules per style
    if style == 'main':
        R = {'front': ('grounds', 0, 'front'), 'p_snow': ('snow', 0, 'entry'), 'p_desert': ('desert', 0, 'entry'),
             'p_lava': ('lava', 0, 'entry'),
             'mirror': [(has('mirror_sw'), ('mirror', 0, 'front')), (None, ('castle', 1, 'mirror'))],
             'basement': ('basement', 0, 'up'), 'p_haunt': ('haunted', 0, 'entry'),
             'p_court': ('courtyard', 0, 'entry'),
             'left': [(visited('lobby'), ('backrooms', 0, 'entry')), (None, ('lobby', 0, 'entry'))],
             'right': [(has('clock_done'), ('observatory', 0, 'entry')), (None, ('clock', 0, 'entry'))],
             'star_door': ('stairs', 0, 'entry'), 'p_sky': ('sky', 0, 'entry'), 'trap': ('basement', 3, 'up'),
             'rainbow': ('rainbow', 0, 'entry')}
    elif style == 'alt':
        R = {'front': ('grounds', 1, 'front'), 'p_snow': ('giant', 0, 'entry'), 'p_desert': ('tiny', 0, 'entry'),
             'p_lava': ('maze', 0, 'entry'), 'mirror': ('castle', 0, 'mirror'), 'basement': ('basement', 5, 'up'),
             'p_haunt': ('haunted', 1, 'entry'), 'p_court': ('aquarium', 0, 'entry'),
             'left': ('lobby', 0, 'd1'), 'right': ('observatory', 0, 'entry'), 'star_door': ('stairs', 0, 'entry'),
             'p_sky': ('rainbow', 0, 'entry'), 'trap': ('city', 0, 'lobby'),
             'rainbow': [(need(30), ('dream', 0, 'entry')), (None, ('sky', 0, 'entry'))]}
    elif style == 'dark':
        R = {'front': ('grounds', 1, 'front'), 'p_snow': ('garden', 1, 'entry'), 'p_desert': ('mono', 0, 'exit'),
             'p_lava': ('lava', 0, 'entry'), 'mirror': ('mirror', 0, 'front'), 'basement': ('basement', 9, 'up'),
             'p_haunt': ('haunted', 1, 'entry'), 'p_court': ('backrooms', 0, 'entry'),
             'left': ('null', 0, 'exit'), 'right': ('clock', 0, 'entry'), 'star_door': ('observatory', 0, 'entry'),
             'p_sky': ('sky', 0, 'entry'), 'trap': ('basement', 8, 'up'), 'rainbow': ('dream', 0, 'entry')}
    else:
        R = {'front': ('castle', 1, 'mirror'), 'p_snow': ('snow', 0, 'entry'), 'p_desert': ('desert', 0, 'entry'),
             'p_lava': ('lava', 0, 'entry'), 'mirror': ('castle', 0, 'mirror'), 'basement': ('basement', 7, 'up'),
             'p_haunt': ('haunted', 0, 'entry'), 'p_court': ('courtyard', 0, 'entry'),
             'left': ('parlor', 0, 'exit'), 'right': ('clock', 0, 'entry'), 'star_door': ('stairs', 0, 'entry'),
             'p_sky': ('sky', 0, 'entry'), 'trap': ('garden', 0, 'entry'), 'rainbow': ('rainbow', 0, 'entry')}
    locks = {}
    if style in ('main', 'mirror'):
        locks['star_door'] = (need(10), "THE STAR DOOR WANTS 10 POWER STARS.")
        locks['p_sky'] = (need(4), "THE PAINTING IS FROZEN. IT WANTS 4 POWER STARS.")
    if style == 'dark':
        locks['rainbow'] = (need(30), "THE RING IS DIM. 30 POWER STARS MIGHT WAKE IT.")
    L = lambda n: locks.get(n)
    ab.portal('front', 0, 0, -16, 0.0, 'door', R['front'], w=3.0, h=4.4, color=fc)
    ab.portal('basement', -20, 0, 10, HALF_PI, 'door', R['basement'], color=fc)
    ab.portal('p_haunt', 20, 0, 10, -HALF_PI, 'door', R['p_haunt'], color=fc)
    ab.portal('p_court', 0, 0, 24, math.pi, 'door', R['p_court'], color=fc)
    ab.portal('left', -12, 6, 24, math.pi, 'door', R['left'], color=fc)
    ab.portal('right', 12, 6, 24, math.pi, 'door', R['right'], color=fc)
    ab.portal('star_door', 0, 6, 24, math.pi, 'door', R['star_door'], w=3.0, h=4.4, lock=L('star_door'),
              color=(240, 210, 60))
    ab.portal('p_snow', -19.7, 1.4, -8, HALF_PI, 'painting', R['p_snow'], fy=0, color=fc)
    ab.portal('p_desert', -19.7, 1.4, 0, HALF_PI, 'painting', R['p_desert'], fy=0, color=fc)
    ab.portal('p_lava', 19.7, 1.4, -8, -HALF_PI, 'painting', R['p_lava'], fy=0, color=fc)
    ab.portal('mirror', 19.7, 1.4, 0, -HALF_PI, 'painting', R['mirror'], fy=0, color=(200, 230, 255))
    ab.portal('p_sky', -6, 7.4, 23.7, math.pi, 'painting', R['p_sky'], fy=6, lock=L('p_sky'), color=fc)
    ab.portal('trap', 0, 0.0, -6, 0.0, 'hole', R['trap'], w=2.0,
              visible=has('carpet') if style == 'main' else None, color=cmul(S['carpet'], 0.6))
    ab.portal('rainbow', 0, 11.2, 11.5, math.pi, 'ring', R['rainbow'], lock=L('rainbow'), fy=8.0)
    ab.spawn('entry', 0, 0, -12, 0.0)
    # power stars / mario npcs / enemies per style (B3313 1.0)
    if style == 'main':
        ab.star(0, 8.9, 11.5)
        ab.star(10, 7.4, 12)
        ab.star(-24, 1.2, 19)
        ab.npc(6, 0, -10, "CASTLE TOAD", ["WELCOME TO THE CASTLE. PLEASE DON'T COUNT THE ROOMS. I STOPPED AT 3000.",
                                        "PAINTINGS ARE DOORS THAT FORGOT HOW TO BE DOORS. JUMP IN.",
                                        "THE WALL BEHIND THE WEST DOORS SOUNDS HOLLOW WHEN NOBODY KNOCKS."],
               style=2, yaw=math.pi)
        ab.trigger(-19, 0, 16, 19, 5, 23.5, 'secret', 'under_balcony', "IT IS QUIET UNDER THE BALCONY.")
    elif style == 'alt':
        ab.star(-10, 7.4, 12)
        ab.star(-24, 1.2, 19)
        ab.npc(6, 0, -10, "TOAD?", ["WELCOME BACK TO THE CASTLE. WAIT. HAVE WE MET BEFORE?",
                                         "THE PAINTINGS SWAPPED PLACES WHILE YOU WERE OUT.",
                                         "THIS IS THE REAL CASTLE. THE OTHER ONE IS THE COPY."],
               style=2, yaw=math.pi)
        ab.enemy('blob', 8, 0.5, -4, 5)
        ab.enemy('blob', -8, 0.5, -10, 5)
    elif style == 'dark':
        ab.star(0, 8.9, 11.5)
        ab.star(-24, 1.2, 19)
        ab.switch(-25.5, 0, 16.5, 'carpet', "A TRAPDOOR OPENS IN A BRIGHTER CASTLE.")
        ab.npc(-6, 0, -10, "BOO", ["THE LIGHTS WENT OUT WHEN SOMEONE COUNTED ALL THE STARS.",
                                     "IN THE BRIGHT CASTLE THE CARPET HIDES A DOOR. HERE, THE CLOSET HIDES THE KEY."],
               style=3, yaw=math.pi)
        ab.enemy('spike', 6, 2, 0, 6)
        ab.enemy('spike', -8, 2, 6, 6)
    else:
        ab.star(10, 7.4, 12)
        ab.star(0, 8.9, 11.5)
        ab.npc(6, 0, -10, "MIRROR TOAD", [".EMOCLEW  ...WELCOME, BACKWARDS.",
                                        "EVERYTHING HERE IS ON THE WRONG SIDE, INCLUDING YOU.",
                                        "THE MIRROR PAINTING LEADS HOME. PROBABLY."], style=2, yaw=math.pi)
    ab.coins_ring(0, 0.8, -4, 4, 8)


def build_basement(ab):
    d = ab.v
    rng = ab.rng
    t = d / 9.0
    wall = lerpc((125, 125, 135), (120, 30, 32), t)
    floor = lerpc((80, 80, 90), (50, 16, 20), t)
    ab.fog = lerpc((50, 52, 62), (40, 5, 10), t)
    ab.sky = (ab.fog, ab.fog, ab.fog)
    ab.view = 70
    ab.fog_near = 0.2
    ab.amb = 0.5
    L = 44 + (d * 13) % 24
    side = d in (4, 6)
    hazard = d in (2, 5, 8)
    alcove = d in (7, 8)
    gaps = {'s': [(0, 2.2, 3.4)], 'n': [(0, 2.2, 3.4)], 'e': [], 'w': []}
    if side:
        gaps['e'].append((L / 2, 2.2, 3.4))
    if alcove:
        gaps['w'].append((L - 6, 2.4, 3.0, 'fake', 'basement_alcove_%d' % d))
    make_room(ab, -5, 0, 5, L, 0, 5, wall, None if hazard else floor, cmul(wall, 0.7), gaps=gaps)
    if hazard:
        g0, g1 = L * 0.38, L * 0.62
        ab.box(-5.3, -1, -0.3, 5.3, 0, g0, floor, tile=4)
        ab.box(-5.3, -1, g1, 5.3, 0, L + 0.3, floor, tile=4)
        ab.box(-5.3, -6, g0, -4.7, 0, g1, wall)
        ab.box(4.7, -6, g0, 5.3, 0, g1, wall)
        ab.lava_plane(-2.5, -5, g0, 5, g1, tile=5)
        n = int((g1 - g0) / 3.2)
        for i in range(n):
            z = g0 + (i + 0.5) * (g1 - g0) / n
            if i % 2 == 1:
                ab.mover(0, -0.3, z, 1.0, 0.3, 0.9, 'line', (3.0, 0, 0), 1.3 + d * 0.1, i, (180, 160, 120))
            else:
                x = rng.uniform(-2.5, 2.5)
                ab.box(x - 1, -3, z - 0.9, x + 1, -0.1, z + 0.9, (130, 120, 110))
        if d == 2:
            ab.star(0, 1.0, (g0 + g1) * 0.5)
        if d == 5:
            ab.box(-4.5, 0, g1 + 3, -2.5, 3.2, g1 + 5, wall)
            ab.star(-3.5, 4.0, g1 + 4)
    if alcove:
        ab.box(-9, -1, L - 8, -5, 0, L - 4, floor)
        wall_x(ab, -9.3, -5, L - 8, 0, 5, wall)
        wall_x(ab, -9.3, -5, L - 4, 0, 5, wall)
        wall_z(ab, L - 8, L - 4, -9, 0, 5, wall)
        ab.box(-9.3, 5, L - 8.3, -5, 5.5, L - 3.7, wall)
        if d == 8:
            ab.star(-7, 1.2, L - 6)
        else:
            ab.switch(-7, 0, L - 6, 'basement_valve', "SOMEWHERE A VALVE TURNS. THE LOOP LOOSENS.")
    if side:
        ab.box(5, -1, L / 2 - 1.5, 5.6, 0, L / 2 + 1.5, floor)
    ab.box(-4.6, 4.2, 0, -3.9, 4.7, L, (90, 90, 95), solid=False)
    ab.box(3.9, 3.8, 0, 4.6, 4.3, L, (130, 80, 60), solid=False)
    for z in range(8, int(L) - 4, 9):
        if hazard and L * 0.36 < z < L * 0.64:
            continue
        for sx in (-3.6, 3.6):
            if rng.random() < 0.7:
                ab.box(sx - .45, 0, z - .45, sx + .45, 5, z + .45, cmul(wall, 0.85))
        ab.box(-0.6, 4.6, z - 0.6, 0.6, 4.75, z + 0.6, (255, 240, 200) if d < 6 else (255, 90, 70), solid=False,
               raw=True, bias=0.2)
    up = ('castle', 0, 'basement') if d == 0 else ('basement', d - 1, 'down')
    if d == 9:
        down = ('city', 0, 'stairs')
    elif d == 7:
        down = [(has('basement_valve'), ('basement', 8, 'up')), (None, ('basement', 3, 'up'))]
    else:
        down = ('basement', d + 1, 'up')
    ab.portal('up', 0, 0, 0, 0.0, 'door', up, color=(150, 150, 150))
    ab.portal('down', 0, 0, L, math.pi, 'door', down, color=(170, 120, 90))
    if side:
        ab.portal('side', 5, 0, L / 2, -HALF_PI, 'door',
                  ('maze', 0, 'basement') if d == 4 else ('aquarium', 0, 'staff'), color=(120, 160, 170))
    if d >= 2:
        for i in range(min(3, 1 + d // 3)):
            z = rng.uniform(6, L - 6)
            if hazard and L * 0.34 < z < L * 0.66:
                z = L * 0.2
            ab.enemy('blob' if d < 6 or i else 'spike', rng.uniform(-3, 3), 0.5, z, 3.5)
    if d == 3:
        ab.npc(2.5, 0, L * 0.5, "BASEMENT TOAD", ["I'VE BEEN MOPPING THIS HALL FOR NINE FLOORS. OR ONE FLOOR NINE TIMES.",
                                                 "FLOOR SEVEN IS A LIAR. IT SENDS YOU BACK UP HERE.",
                                                 "THERE'S A VALVE BEHIND A WALL THAT ISN'T. DOWN ON SEVEN."],
               style=4, yaw=-HALF_PI)
    ab.coins_line((0, 0.8, 4), (0, 0.8, 12), 5)


def build_courtyard(ab):
    ab.sky = ((70, 150, 235), (200, 235, 255), (90, 140, 170))
    ab.fog = (205, 230, 250)
    sw = ab.w.has('court_sw')
    stone = (205, 200, 185)
    ab.box(-25, -5, -25, 25, -4, 25, (120, 140, 150), tile=6)
    ab.box(-25.3, -4, -25.3, 25.3, 0, -20, stone, top=(220, 215, 200))
    ab.box(-25.3, -4, 20, 25.3, 0, 25.3, stone, top=(220, 215, 200))
    ab.box(-25.3, -4, -20, -20, 0, 20, stone, top=(220, 215, 200))
    ab.box(20, -4, -20, 25.3, 0, 20, stone, top=(220, 215, 200))
    wallc = (190, 170, 150)
    wall_x(ab, -25.6, 25.6, -25.3, 0, 10, wallc, [(0, 2.2, 3.4)])
    wall_x(ab, -25.6, 25.6, 25.3, 0, 10, wallc, [(0, 2.2, 3.4)])
    wall_z(ab, -25, 25, -25.3, 0, 10, wallc)
    wall_z(ab, -25, 25, 25.3, 0, 10, wallc)
    if not sw:
        for i in range(5):
            x = -0.9 + i * 0.45
            ab.box(x - .08, 0, 25.0, x + .08, 3.4, 25.2, (60, 60, 70))
    ab.water_plane(-0.6, -20, -20, 20, 20, (60, 140, 210))
    ab.prism(0, 0, -4, 0.4, 4, 10, stone, top=(120, 170, 220))
    ab.prism(0, 0, 0.4, 2.4, 1.6, 8, stone)
    ab.prism(0, 0, 2.4, 4.4, 0.8, 8, stone, top=(240, 230, 210))
    ab.star(0, 5.3, 0)
    for i in range(6):
        a = TAU * i / 6 + 0.3
        ab.prism(math.cos(a) * 12, math.sin(a) * 12, -4, 0.3, 1.2, 7, (170, 165, 150))
    for x in (-15, -5, 5, 15):
        make_arch(ab, x, -22.5, 0, 3, 5, 1.2, 0, (180, 160, 140))
        make_arch(ab, x, 22.5, 0, 3, 5, 1.2, 0, (180, 160, 140))
    ab.box(19, 0, 19, 24.8, 2, 24.8, (160, 150, 140))
    ab.box(21.5, 2, 21.5, 24.8, 4, 24.8, (150, 140, 130))
    ab.switch(23, 4, 23, 'court_sw', "THE NORTH GATE UNLOCKS WITH A SIGH.")
    ab.star(-12, 0.9, -12, cond=has('court_sw'))
    ab.portal('entry', 0, 0, -25.3, 0.0, 'door', ('castle', 0, 'p_court'))
    ab.portal('drain', 14, -4, 14, 0.0, 'hole', ('aquarium', 0, 'drain'), w=2.4, color=(40, 60, 80))
    ab.portal('gate', 0, 0, 25.3, math.pi, 'door', ('garden', 0, 'gate'), visible=has('court_sw'),
              color=(120, 200, 120))
    ab.enemy('spike', 0, 3, -10, 6)
    ab.npc(-22, 0, -22, "COURTYARD TOAD", ["THE WATER GOES SOMEWHERE WHEN IT DRAINS. SO COULD YOU.",
                                            "THE GATE NORTH IS SHUT. THE SWITCH IS UP IN THE CORNER. OBVIOUSLY."],
           style=1, yaw=HALF_PI / 2)
    ab.coins_ring(0, 1.0, 0, 8, 10)
    ab.spawn('entry', 0, 0, -22, 0.0)


def build_aquarium(ab):
    ab.fog = (20, 60, 70)
    ab.sky = (ab.fog, ab.fog, ab.fog)
    ab.amb = 0.5
    ab.view = 90
    wall = (70, 110, 120)
    floor = (60, 80, 90)
    make_room(ab, -12, 0, 12, 80, 0, 10, wall, None, (40, 70, 80),
              gaps={'s': [(0, 2.2, 3.4)], 'n': [(0, 2.2, 3.4)]})
    ab.box(-12.3, -1, -0.3, 12.3, 0, 30, floor, checker=(70, 95, 105), tile=4)
    ab.box(-12, -7, 30, 12, -6, 50, (40, 60, 70))
    ab.box(-12.3, -7, 30, -11.7, 0, 50, wall)
    ab.box(11.7, -7, 30, 12.3, 0, 50, wall)
    ab.box(-12.3, -1, 50, 12.3, 0, 80.3, floor, checker=(70, 95, 105), tile=4)
    ab.box(-12, -7, 29.7, 12, -1, 30, wall)
    ab.box(-12, -7, 50, 12, -1, 50.3, wall)
    ab.water_plane(-1.2, -12, 30, 12, 50, (30, 110, 140))
    glass = (60, 150, 170)
    for z in (6, 18, 56, 68):
        ab.box(-12, 0, z, -7, 6, z + 8, glass, top=(40, 120, 150))
        ab.box(7, 0, z, 12, 6, z + 8, glass, top=(40, 120, 150))
        for k in range(3):
            fz = z + 2 + k * 2
            ab.box(-10.5, 1.5 + k, fz, -9.8, 2.0 + k, fz + 1.0, hsv(0.08 * k + z * 0.01, 0.8, 1.0), solid=False,
                   raw=True, bias=-0.2)
    make_stairs(ab, -6, 1, 0, 15, 0, 1, 2.0, 0.4, 0.34, (110, 110, 100))
    ab.star(-9.5, 6.9, 22)
    ab.star(0, -5.1, 40)
    ab.switch(8, 0, 77, 'mirror_sw', "A MIRROR SOMEWHERE STOPS REFLECTING.")
    ab.portal('entry', 0, 0, 0, 0.0, 'door', ('castle', 1, 'p_court'))
    ab.portal('staff', 0, 0, 80, math.pi, 'door', ('basement', 6, 'side'), color=(120, 120, 130))
    ab.portal('drain', 8, 1.2, 4, 0.0, 'pipe', ('courtyard', 0, 'drain'))
    ab.npc(-3, 0, 56, "PENGUIN CURATOR", ["THE FISH LEFT YEARS AGO. THEY TOOK THE LABELS WITH THEM.",
                                  "THE SWITCH AT THE FAR END TURNS OFF A MIRROR. DON'T ASK WHICH ONE.",
                                  "SOMETHING SHINY SANK TO THE BOTTOM OF THE PIT."], style=4, yaw=0.0)
    ab.enemy('blob', 0, 0.5, 62, 5)
    ab.enemy('spike', 0, -3, 40, 5)
    ab.coins_line((0, 0.8, 52), (0, 0.8, 74), 6)


def build_snow(ab):
    seed = ab.w.seed
    rng = ab.rng
    ab.sky = ((110, 150, 220), (230, 240, 255), (200, 210, 230))
    ab.fog = (225, 235, 250)

    def H(x, z):
        d = math.hypot(x, z)
        m = max(0.0, 38 - d * 0.62)
        m = min(m, 36.0)
        n = fbm(x * 0.05, z * 0.05, seed + 3, 3) * 4.0
        h = m + n * smooth(d / 8)
        dl = math.hypot(x + 45, z + 30)
        if dl < 16:
            h = lerp(h, -2.0, 1 - smooth((dl - 10) / 6))
        r = max(abs(x), abs(z))
        if r > 66:
            h += (r - 66) * 0.7
        return h

    def cfn(x, z, h, i, j):
        if h < -1:
            return (170, 190, 220)
        k = vnoise(x * 0.3, z * 0.3, seed)
        if h > 20 and k > 0.6:
            return (150, 150, 165)
        return lerpc((225, 230, 245), (250, 250, 255), k)

    ab.heightfield(-80, -80, 5, 32, 32, H, cfn)
    for (a, b, c, d) in ((-81, -81, -80, 81), (80, -81, 81, 81), (-81, -81, 81, -80), (-81, 80, 81, 81)):
        ab.box(a, -10, b, c, 70, d, (0, 0, 0), visible=False)
    top = H(0, 0)
    ab.box(-4, top - 2, -4, 4, top + 0.3, 4, (200, 210, 230), top=(250, 250, 255))
    ab.star(2.5, top + 1.4, 2.5)
    ab.portal('summit', 0, top + 3.8, -1.5, 0.0, 'ring', ('sky', 0, 'snow'), fy=top + 0.3)
    ab.water_plane(-0.8, -60, -45, -30, -15, (150, 190, 230))
    for i, (x, z) in enumerate(((-38, -24), (-42, -29), (-47, -33), (-51, -29))):
        ab.box(x - 1.3, -1.5, z - 1.3, x + 1.3, -0.35, z + 1.3, (235, 245, 255))
    ab.star(-51, 0.7, -29)
    tg = H(40, 30)
    make_tower(ab, 40, 30, tg - 1, 15, 3.0, (170, 200, 230), top=(220, 240, 255))
    spiral_steps(ab, 40, 30, tg, 13, 4.9, 0.6, 1.1, 1.0, (200, 225, 245))
    ab.star(40, tg + 15.0, 30)
    for i in range(24):
        x, z = rng.uniform(-70, 70), rng.uniform(-70, 70)
        if math.hypot(x, z) < 12 or math.hypot(x + 45, z + 30) < 18 or math.hypot(x - 40, z - 30) < 8 or \
                (abs(x) < 6 and z < -55):
            continue
        make_tree(ab, x, z, H(x, z), rng.uniform(1.0, 1.6), trunk=(90, 70, 60), leaf=(40, 100, 80), kind='pine')
    ab.spawn('entry', 0, H(0, -56) + 0.2, -56, 0.0)
    ab.portal('entry', 0, H(0, -65) + 2.2, -65, 0.0, 'ring', ('castle', 0, 'p_snow'), fy=H(0, -63))
    cg = H(18, -24)
    ab.portal('cave', 18, cg, -24, math.pi, 'door', ('caverns', 0, 'snow'), free=True, color=(120, 110, 140))
    for (x, z) in ((10, -45), (-20, -10), (25, 10)):
        ab.enemy('blob', x, H(x, z) + 1, z, 6, col=(235, 240, 255))
    g = H(-12, -58)
    ab.npc(-12, g, -58, "PENGUIN", ["I'M NOT A SNOWMAN. I'M A MAN WHO IS VERY COLD.",
                                    "THE SUMMIT RING GOES UP, NOT DOWN. UP IS WHERE THE ISLANDS ARE.",
                                    "THERE'S A POWER STAR ON THE FROZEN LAKE. HOP THE ICE."], style=5, yaw=0.3)
    ab.coins_line((0, H(0, -60) + 1, -60), (0, H(0, -30) + 1, -30), 6)


def build_lava(ab):
    rng = ab.rng
    ab.sky = ((40, 5, 5), (160, 50, 20), (60, 10, 5))
    ab.fog = (95, 28, 16)
    ab.amb = 0.55
    ab.kill_y = -20
    ab.lava_plane(-2, -44, -24, 44, 124, tile=11)
    metal = (120, 115, 125)
    ab.box(-6, -4, -17, 6, 0, 6, metal, top=(150, 145, 150))
    ab.portal('entry', 0, 2.3, -14, 0.0, 'ring', ('castle', 0, 'p_lava'), fy=0)
    ab.spawn('entry', 0, 0, -1, 0.0)
    ab.box(-1.5, -4, 8, 1.5, 0.5, 12, metal)
    ab.mover(0, 0.4, 16.5, 1.5, 0.3, 1.5, 'line', (5.0, 0, 0), 1.2, 0.0, (200, 170, 90))
    ab.box(-2, -4, 21, 2, 0.8, 25, metal)
    for i in range(3):
        ab.mover(-4 + i * 4, 0.5, 29 + i * 3, 1.3, 0.4, 1.3, 'line', (0, 1.5, 0), 1.6, i * 1.3, (180, 120, 80))
    ab.box(-10, -4, 38, 10, 1, 52, metal, top=(140, 135, 145), tile=5)
    make_tower(ab, 0, 45, 1, 20, 3.0, (90, 80, 85), crenel=True)
    ab.mover(5.2, 10.8, 45, 1.3, 0.3, 1.3, 'line', (0, 9.6, 0), 0.55, 0.0, (230, 200, 60))
    ab.star(0, 22.2, 45)
    ab.enemy('cube', -6, 1.6, 42, 5, axis=0)
    zc = 56
    x = 0.0
    for i in range(9):
        x = clamp(x + rng.uniform(-4, 4), -12, 12)
        if i % 3 == 2:
            ab.mover(x, 0.5, zc, 1.5, 0.35, 1.5, 'line', (0, 0, 2.0), 1.4, i, (200, 150, 90))
        else:
            h = rng.uniform(0.2, 1.6)
            ab.box(x - 1.5, -4, zc - 1.5, x + 1.5, h, zc + 1.5, metal)
        zc += 4.2
    ab.box(-8, -4, 94, 8, 1, 108, metal, top=(140, 135, 145), tile=5)
    ab.portal('vent', 4, 2.2, 104, math.pi, 'pipe', ('maze', 0, 'lava'))
    ab.star(-4, 2.2, 104)
    ab.enemy('blob', 0, 1.5, 100, 4)
    for z in range(0, 120, 16):
        for sx in (-30, 30):
            ab.prism(sx, z, -2, 26, 2.5, 8, (70, 60, 65), solid=False)
            ab.prism(sx, z, 26, 27, 2.9, 8, (255, 120, 40), solid=False, raw=True)
    ab.npc(4, 1, 48, "BOB-OMB BUDDY", ["IT'S WARM HERE. DO YOU LIKE WARM? I LIKE WARM.",
                                      "THE LIFT GOES TO THE CHIMNEY. THE CHIMNEY GOES TO THE STARS. ONE OF THEM."],
           style=3, yaw=-HALF_PI)
    ab.coins_line((0, 1.4, 8), (0, 1.4, 24), 5)


def build_desert(ab):
    seed = ab.w.seed
    ab.sky = ((80, 140, 230), (250, 220, 170), (200, 160, 110))
    ab.fog = (240, 215, 170)

    def H(x, z):
        h = fbm(x * 0.03, z * 0.03, seed + 9, 3) * 7 + math.sin(x * 0.09 + z * 0.03) * 1.5 - 2
        d = math.hypot(x, z - 40)
        h *= smooth((d - 20) / 8)
        do = math.hypot(x + 40, z + 20)
        if do < 12:
            h = lerp(h, -2.6, 1 - smooth((do - 7) / 5))
        r = max(abs(x), abs(z - 10))
        if r > 64:
            h += (r - 64) * 0.7
        return h

    def cfn(x, z, h, i, j):
        k = vnoise(x * 0.15, z * 0.15, seed + 1)
        if h < -1.5:
            return (160, 150, 90)
        return lerpc((225, 190, 120), (245, 215, 150), k)

    ab.heightfield(-80, -60, 5, 32, 32, H, cfn)
    for (a, b, c, d) in ((-81, -61, -80, 101), (80, -61, 81, 101), (-81, -61, 81, -60), (-81, 100, 81, 101)):
        ab.box(a, -10, b, c, 60, d, (0, 0, 0), visible=False)
    sand = (220, 185, 120)
    for i in range(8):
        hh = 15 - i * 1.8
        ab.box(-hh, i * 1.8 - (1 if i == 0 else 0), 40 - hh, hh, (i + 1) * 1.8, 40 + hh, cmul(sand, 1 - i * 0.03))
    ab.portal('pyramid', 0, 14.4, 40, math.pi, 'hole', ('city', 0, 'pyramid'), w=2.0, color=(80, 60, 30))
    ab.star(3.4, 13.5, 43.4)
    ab.portal('tomb', 0, 0, 24.9, math.pi, 'door', ('maze', 0, 'tomb'), color=(120, 90, 50))
    ab.fake_wall(-1.6, 0, 23.6, 1.6, 3.8, 24.95, cmul(sand, 1.02), 'desert_tomb', bias=-0.3)
    # ruins ring
    cx, cz = 38, -12
    heights = [2.0, 3.8, 5.6, 7.4, 9.2, 4.0, 1.5, 6.0, 3.0, 2.5]
    for i, hh in enumerate(heights):
        a = TAU * i / len(heights)
        px, pz = cx + math.cos(a) * 9, cz + math.sin(a) * 9
        g = H(px, pz)
        ab.prism(px, pz, g - 1, g + hh, 1.3, 7, (210, 195, 170), top=(230, 215, 190))
        if i == 4:
            ab.star(px, g + hh + 0.9, pz)
    make_arch(ab, cx, cz + 16, H(cx, cz + 16), 4, 5, 1.5, 0, (200, 180, 150))
    ab.water_plane(-1.4, -52, -32, -28, -8, (60, 160, 190))
    for (x, z) in ((-33, -12), (-47, -26), (-34, -28), (-48, -13)):
        make_tree(ab, x, z, H(x, z), 1.2, trunk=(150, 110, 60), leaf=(60, 150, 60), kind='palm')
    ab.spawn('entry', 0, H(0, -48) + 0.2, -48, 0.0)
    ab.portal('entry', 0, H(0, -58) + 2.2, -58, 0.0, 'ring', ('castle', 0, 'p_desert'), fy=H(0, -56))
    for (x, z) in ((15, 0), (-20, 10)):
        ab.enemy('blob', x, H(x, z) + 1, z, 6, col=(200, 140, 70))
    ab.enemy('spike', 30, H(30, 5) + 2, 5, 6)
    ab.npc(-8, H(-8, -40), -40, "LAKITU", ["THE PYRAMID IS HOLLOW. THE TOP IS A DOOR IF YOU'RE BRAVE ENOUGH TO FALL.",
                                                  "ONE STONE AT THE PYRAMID'S BASE IS JUST PRETENDING.",
                                                  "THE COLUMNS OUT EAST CLIMB LIKE STAIRS FOR GIANTS."], style=4,
           yaw=0.5)
    ab.coins_ring(38, H(38, -12) + 1.0, -12, 4, 8)


def build_haunted(ab):
    other = ab.v == 1
    wall = (200, 190, 220) if other else (70, 50, 62)
    floor = (230, 220, 200) if other else (62, 40, 32)
    ab.fog = (220, 215, 235) if other else (30, 20, 42)
    ab.sky = (ab.fog, ab.fog, ab.fog)
    ab.amb = 0.7 if other else 0.45
    ab.view = 110
    make_room(ab, -15, 0, 15, 30, 0, 12, wall, floor, cmul(wall, 0.7), checker=cmul(floor, 1.15),
              gaps={'s': [(0, 2.2, 3.4)], 'w': [(15, 2.2, 3.4)],
                    'e': [(15, 2.2, 3.4), (5, 2.4, 2.6, 'fake', 'haunt_fireplace')], 'n': None})
    wall_x(ab, -15.3, 15.3, 30, 0, 6, wall)
    wall_x(ab, -15.3, 15.3, 30, 6, 12, wall, [(0, 2.2, 3.4)])
    ab.box(-14.7, 5.4, 22, 14.7, 6, 29.7, cmul(floor, 0.9))
    make_stairs(ab, 10, 8, 0, 15, 0, 1, 3, 0.4, 14 / 15, cmul(floor, 1.2))
    shelf = (230, 200, 230) if other else (90, 60, 40)
    ab.box(-14.7, 0, 16, -11, 7, 20, shelf)
    for k in range(4):
        ab.box(-11.05, 1 + k * 1.5, 16.2, -10.95, 1.8 + k * 1.5, 19.8, hsv(0.1 * k + 0.5 * other, 0.6, 0.7),
               solid=False, bias=0.3)
    ab.box(-5, 0, 8, 5, 1.2, 12, shelf)
    for x in (-4, 0, 4):
        ab.box(x - .6, 0, 6.2, x + .6, 0.7, 7.4, shelf)
        ab.box(x - .6, 0, 12.6, x + .6, 0.7, 13.8, shelf)
    ab.box(14.4, 0, 3.2, 15, 3, 6.8, (40, 20, 20), solid=False, bias=0.2)
    ab.box(15, -1, 3, 19, 0, 7, floor)
    wall_x(ab, 15, 19.3, 3, 0, 4, wall)
    wall_x(ab, 15, 19.3, 7, 0, 4, wall)
    wall_z(ab, 3, 7, 19, 0, 4, wall)
    ab.box(15, 4, 2.7, 19.3, 4.5, 7.3, wall)
    if other:
        ab.prism(0, 17.5, 6.7, 7.2, 1.5, 8, (240, 230, 200), bottom=True)
        ab.star(0, 8.0, 17.5)
    else:
        ab.star(-12.8, 7.9, 18)
        ab.star(17, 1.2, 5)
    R = {'entry': ('castle', 1 if other else 0, 'p_haunt'),
         'library': ('haunted', 0 if other else 1, 'library'),
         'attic': ('dark', 0, 'front') if other else ('haunted', 0, 'cellar'),
         'cellar': ('mirror', 0, 'front') if other else ('haunted', 0, 'attic')}
    fc = (200, 200, 240) if other else (120, 80, 60)
    ab.portal('entry', 0, 0, 0, 0.0, 'door', R['entry'], color=fc)
    ab.portal('library', -15, 0, 15, HALF_PI, 'door', R['library'], color=fc)
    ab.portal('cellar', 15, 0, 15, -HALF_PI, 'door', R['cellar'], color=fc)
    ab.portal('attic', 0, 6, 30, math.pi, 'door', R['attic'], color=fc)
    ab.enemy('spike', -6, 2.5, 20, 5, col=(200, 120, 255))
    ab.enemy('spike', 6, 2.5, 4, 5, col=(200, 120, 255))
    if other:
        ab.npc(-8, 0, 4, "HAUNTED TOAD", ["THIS MANSION IS THE BRIGHT ONE. THE DARK ONE IS A RUMOUR I STARTED.",
                                              "THE ATTIC HERE OPENS ONTO A CASTLE WITH NO LIGHTS."], style=2,
               yaw=0.4)
    else:
        ab.npc(-8, 0, 4, "MANSION TOAD", ["THE ATTIC LEADS TO THE CELLAR. THE CELLAR LEADS TO THE ATTIC. I LEAD NOWHERE.",
                                        "THE LIBRARY DOOR OPENS INTO A HOUSE THAT IS THIS HOUSE, INVERTED.",
                                        "THE FIREPLACE HASN'T BEEN LIT IN YEARS. IT ISN'T EVEN REAL."], style=2,
               yaw=0.4)
    ab.coins_line((-10, 0.8, 24), (10, 0.8, 24), 5)


def build_clock(ab):
    ab.fog = (70, 52, 34)
    ab.sky = (ab.fog, ab.fog, ab.fog)
    ab.amb = 0.55
    ab.view = 130
    R = 18.0
    H = 64.0
    brass = (190, 150, 80)
    ab.prism(0, 0, 0, H, R, 16, (120, 90, 60), inward=True, cap=False,
             colr=[(120, 90, 60), (110, 82, 55)])
    ab.prism(0, 0, -1, 0, R + 0.5, 16, (90, 70, 50), top=(140, 110, 70))
    ab.prism(0, 0, H, H + 1, R + 0.5, 16, (60, 45, 30), bottom=True, cap=False)
    ab.prism(0, 0, 0, 55, 2.2, 10, brass, top=(230, 190, 100))
    a0 = -HALF_PI + 0.45
    last = spiral_steps(ab, 0, 0, 0.0, 46, R - 2.4, 0.19, 1.2, a0, brass,
                        size=1.2, step_fn=lambda i: (200, 160, 90) if i % 2 else (170, 130, 70))
    lx, ly, lz, la = last
    for k in range(1, 5):
        rr = R - 2.4 - k * 2.6
        px, pz = math.cos(la) * rr, math.sin(la) * rr
        ab.box(px - 0.9, ly - 0.5, pz - 0.9, px + 0.9, ly, pz + 0.9, (230, 190, 90))
    ab.star(0, 56.2, 0)
    ab.portal('top', 0, 59.0, 0, 0.0, 'ring', ('castle', 0, 'right'), fy=55.0)
    for i, y in enumerate((12.0, 24.0, 36.0)):
        ab.mover(0, y, 0, 1.5, 0.25, 1.5, 'circle', (9.0, 0, 0), 0.45 * (1 if i % 2 else -1), i * 2.0,
                 (220, 180, 90))
    px, pz = math.cos(1.0) * 4.0, math.sin(1.0) * 4.0
    ab.box(px - 1.2, 23.8, pz - 1.2, px + 1.2, 24.3, pz + 1.2, (240, 200, 100))
    ab.star(px, 25.2, pz)
    ab.mover(0, 3.0, 8.0, 1.6, 0.3, 1.6, 'line', (9.0, 0, 0), 1.0, 0.0, (160, 160, 170))
    fa = a0 + 20 * 0.19
    frr = R - 2.4
    ab.portal('face', math.cos(fa) * frr, 1.2 * 21 + 2.6, math.sin(fa) * frr,
              math.atan2(-math.cos(fa), -math.sin(fa)), 'ring', ('observatory', 0, 'clock'), fy=1.2 * 21)
    for k in range(12):
        a = TAU * k / 12
        px, pz = math.cos(a) * (R - 0.6), math.sin(a) * (R - 0.6)
        ab.box(px - .3, 44 + math.sin(a) * 6 - 1.2, pz - .3, px + .3, 44 + math.sin(a) * 6 + 1.2, pz + .3,
               (250, 240, 200), solid=False, raw=True)
    ab.portal('entry', 0, 0, -R + 1.2, 0.0, 'door', ('castle', 0, 'right'), free=True, color=brass)
    ab.npc(4, 0, -12, "CLOCK TOAD", ["EVERY GEAR HERE TURNS A DIFFERENT DAY OF THE WEEK.",
                               "REACH THE TOP AND THE CASTLE WILL REMEMBER YOU WERE HERE. DOORS CHANGE FOR THAT.",
                               "HALFWAY UP THERE'S A RING THAT TICKS TOWARD THE STARS."], style=4, yaw=-0.4)
    ab.coins_ring(0, 0.8, 0, 6, 10)
    ab.spawn('entry', 0, 0, -7.5, 0.0)


def build_sky(ab):
    rng = ab.rng
    ab.sky = ((90, 140, 250), (250, 220, 250), (160, 190, 255))
    ab.fog = (225, 228, 255)
    ab.kill_y = -40
    make_island(ab, 0, 0, 0, 8)
    ab.portal('entry', 0, 2.3, -6.6, 0.0, 'ring', ('castle', 0, 'p_sky'), fy=0)
    ab.portal('down', 4.5, 0.0, 3.5, 0.0, 'hole', ('grounds', 0, 'tower'), w=2.0, color=(100, 160, 90))
    ab.spawn('entry', 0, 0, 2.5, 0.0)
    isl = [(0.0, 0.0, 0.0, 8.0)]
    heading = 0.0
    x = y = z = 0.0
    r0 = 8.0
    for i in range(12):
        heading += rng.uniform(-0.7, 0.7)
        r1 = rng.uniform(3.2, 5.5)
        dist = r0 + r1 + rng.uniform(1.8, 4.2)
        x += math.sin(heading) * dist
        z += math.cos(heading) * dist
        y += rng.choice((0.0, 0.8, 1.2, 1.6, -1.0))
        make_island(ab, x, y, z, r1, top=hsv(0.25 + i * 0.03, 0.55, 0.85), rot=i)
        isl.append((x, y, z, r1))
        if rng.random() < 0.35:
            make_tree(ab, x + r1 * 0.3, z, y, 0.9, leaf=(90, 200, 120))
        r0 = r1
    a = isl[4]
    ab.star(a[0], a[1] + 1.2, a[2])
    b = isl[7]
    ab.star(b[0], b[1] + 4.0, b[2])
    c = isl[-1]
    ab.portal('rainbow', c[0], c[1] + 2.3, c[2], 0.0, 'ring', ('rainbow', 0, 'entry'), fy=c[1])
    ab.star(c[0] + 2, c[1] + 1.2, c[2] + 1.5)
    d = isl[5]
    ab.portal('snow', d[0], d[1] + 2.3, d[2] + 1, 0.0, 'ring', ('snow', 0, 'summit'), fy=d[1])
    e = isl[2]
    ab.npc(e[0], e[1], e[2], "CLOUD LAKITU", ["I CAME UP HERE TO GET AWAY FROM THE DOORS. THEN THE DOORS CAME UP TOO.",
                                              "ONE ISLAND HAS A POWER STAR FLOATING HIGH. JUMP THREE TIMES, LIKE A SONG."],
           style=5, yaw=0.0)
    for i in range(2):
        p = isl[3 + i * 5]
        ab.enemy('spike', p[0], p[1] + 2.2, p[2], 3.5)
    for i in range(14):
        cx, cz = rng.uniform(-80, 80), rng.uniform(-40, 120)
        cy = rng.uniform(-22, -12)
        w_ = rng.uniform(6, 14)
        ab.box(cx - w_, cy, cz - w_ * 0.6, cx + w_, cy + 2, cz + w_ * 0.6, (250, 250, 255), solid=False, raw=True)
    for i in range(1, len(isl) - 1, 3):
        p = isl[i]
        ab.coins_ring(p[0], p[1] + 0.9, p[2], max(1.5, p[3] - 1.5), 5)


def build_caverns(ab):
    seed = ab.w.seed
    rng = ab.rng
    ab.sky = ((10, 8, 20), (30, 25, 40), (10, 8, 20))
    ab.fog = (26, 22, 38)
    ab.amb = 0.45
    ab.view = 95
    ab.fog_near = 0.25

    def H(x, z):
        h = fbm(x * 0.06, z * 0.06, seed + 21, 3) * 5 - 1
        d = math.hypot(x - 20, z - 20)
        if d < 14:
            h = lerp(h, -4.5, 1 - smooth((d - 8) / 6))
        return h

    def cfn(x, z, h, i, j):
        k = vnoise(x * 0.3, z * 0.3, seed + 4)
        return lerpc((80, 70, 75), (110, 95, 90), k)

    ab.heightfield(-52, -52, 4, 26, 26, H, cfn)
    ab.qgrid((-52, 16, -52), (52, 16, -52), (52, 16, 52), (-52, 16, 52), 9, 9, (55, 45, 55), (0, -1, 0),
             col2=(48, 40, 50))
    ab.col_box(-52, 16, -52, 52, 20, 52)
    rock = (90, 75, 80)
    for (a, b, c, d) in ((-53, -53, -52, 53), (52, -53, 53, 53), (-53, -53, 53, -52), (-53, 52, 53, 53)):
        ab.box(a, -10, b, c, 16, d, rock, tile=8)
    ab.water_plane(-2.2, 8, 8, 32, 32, (30, 80, 120))
    for i in range(22):
        x, z = rng.uniform(-48, 48), rng.uniform(-48, 48)
        if math.hypot(x - 20, z - 20) < 15 or math.hypot(x + 30, z - 30) < 7:
            continue
        g = H(x, z)
        h = rng.uniform(2, 6)
        ab.prism(x, z, g - 0.5, g + h, rng.uniform(0.8, 1.6), 6, rock, rt=0.2, rot=rng.random())
        x2, z2 = rng.uniform(-48, 48), rng.uniform(-48, 48)
        ab.prism(x2, z2, 16 - rng.uniform(2, 5), 16, 0.0, 6, (70, 60, 70), rt=rng.uniform(0.7, 1.4), solid=False,
                 cap=False)
    for i in range(14):
        x, z = rng.uniform(-48, 48), rng.uniform(-48, 48)
        g = H(x, z)
        c = hsv(rng.choice((0.5, 0.8, 0.55)), 0.7, 1.0)
        ab.prism(x, z, g - 0.2, g + rng.uniform(0.8, 1.8), 0.35, 4, c, rt=0.0, solid=False, raw=True)
    pg = H(-30, 30)
    ab.prism(-30, 30, pg - 1, pg + 12, 2.5, 8, (100, 85, 90))
    spiral_steps(ab, -30, 30, pg, 11, 4.3, 0.65, 1.05, 0.0, (120, 100, 105))
    ab.star(-30, pg + 13.0, 30)
    ab.star(20, -3.6, 20)
    ab.portal('pipe', -40, H(-40, -40) + 1.2, -40, 0.0, 'pipe', ('grounds', 0, 'pipe'))
    ab.portal('snow', 40, H(40, -40), -40, -HALF_PI, 'door', ('snow', 0, 'cave'), free=True, color=(160, 170, 200))
    ab.portal('deep', 0, H(0, 40), 40, 0.0, 'hole', ('city', 0, 'caves'), w=2.2, color=(60, 50, 60))
    ab.spawn('entry', -40, H(-40, -36) + 0.2, -36, 0.0)
    for (x, z) in ((-10, -20), (10, 0), (-20, 10)):
        ab.enemy('blob', x, H(x, z) + 1, z, 6, col=(120, 110, 150))
    ab.npc(-36, H(-36, -30), -30, "MONTY MOLE", ["I'VE MAPPED THESE TUNNELS 40 TIMES. THEY'VE DISAGREED 40 TIMES.",
                                                   "THE HOLE TO THE NORTH GOES DOWN TO A CITY THAT FORGOT THE SUN.",
                                                   "SHINY THINGS SINK. CHECK THE LAKE."], style=4, yaw=0.8)
    ab.coins_line((-40, H(-40, -30) + 1, -30), (-10, H(-10, -30) + 1, -30), 6)


def build_rainbow(ab):
    rng = ab.rng
    ab.starfield = True
    ab.sky = ((10, 5, 40), (110, 60, 160), (20, 10, 40))
    ab.fog = (70, 40, 110)
    ab.kill_y = -40
    ab.hue_cycle = True
    ab.box(-4, -1, -11, 4, 0, 4, (240, 240, 255))
    ab.portal('entry', 0, 2.3, -9, 0.0, 'ring', ('castle', 0, 'rainbow'), fy=0)
    ab.spawn('entry', 0, 0, 1, 0.0)
    x = y = 0.0
    z = 4.0
    heading = 0.0
    pts = []
    for i in range(26):
        heading = clamp(heading + rng.uniform(-0.6, 0.6), -1.0, 1.0)
        step = rng.uniform(3.8, 5.6)
        x += math.sin(heading) * step
        z += math.cos(heading) * step
        y += rng.choice((0.0, 0.5, 1.0, 1.4))
        c = hsv(i / 26.0, 0.75, 1.0)
        if i % 6 == 5:
            ab.mover(x, y, z, 1.4, 0.3, 1.4, 'line', (2.5, 0, 0), 1.1, i, c)
        else:
            s = 1.6 if i % 4 else 1.1
            ab.box(x - s, y - 0.6, z - s, x + s, y, z + s, c, raw=True)
        pts.append((x, y, z))
    bx, by, bz = pts[12]
    for k in range(5):
        bx += 4.3
        by += 0.6
        ab.box(bx - 1.2, by - 0.5, bz - 1.2, bx + 1.2, by, bz + 1.2, hsv(0.9 - k * 0.05, 0.4, 1.0), raw=True)
    ab.star(bx, by + 1.2, bz)
    ex, ey, ez = pts[-1]
    ez += 7
    ab.box(ex - 5, ey - 1, ez - 5, ex + 5, ey, ez + 5, (250, 250, 250), checker=(255, 200, 240))
    ab.star(ex - 3, ey + 1.2, ez)
    ab.portal('end', ex, ey + 2.3, ez + 3, 0.0, 'ring',
              [(need(30), ('dream', 0, 'entry')), (None, ('sky', 0, 'rainbow'))], fy=ey)
    ab.enemy('spike', pts[8][0], pts[8][1] + 2, pts[8][2], 2.5)
    ab.enemy('spike', pts[18][0], pts[18][1] + 2, pts[18][2], 2.5)
    ab.npc(ex + 3, ey, ez - 2, "YOSHI", ["EVERY COLOUR IS A FLOOR IF YOU BELIEVE HARD ENOUGH.",
                                         "WITH THIRTY POWER STARS THE RING HERE STOPS BEING POLITE AND STARTS BEING A DOOR."],
           style=5, yaw=math.pi)
    for i in range(0, 26, 4):
        p = pts[i]
        ab.coin(p[0], p[1] + 1.0, p[2])


def build_tiny(ab):
    ab.sky = ((255, 190, 220), (255, 240, 220), (220, 200, 240))
    ab.fog = (255, 235, 235)
    ab.kill_y = -20
    s = 0.25
    ab.s = s
    ab.suppress = True
    build_grounds(ab, replica=True)
    ab.s = 1.0
    ab.suppress = False
    seed = ab.w.seed
    g = lambda x, z: s * grounds_h(seed, x / s, z / s)
    ab.star(0, 14 * s + 0.9, 50 * s)
    ab.star(42 * s, g(42 * s, -22 * s) + 12 * s + 0.9, -22 * s)
    ab.spawn('entry', 0, g(0, -7) + 0.2, -7, 0.0)
    ab.portal('entry', 0, g(0, -14.5) + 2.2, -14.5, 0.0, 'ring', ('grounds', 0, 'tree'), fy=g(0, -13))
    ab.portal('twin', -7, g(-7, -4) + 1.2, -4, 0.0, 'pipe', ('giant', 0, 'twin'))
    ab.npc(5, g(5, -9), -9, "TINY TOAD", ["YOU'RE ENORMOUS. PLEASE MIND THE CASTLE, WE JUST HAD IT PAINTED.",
                                           "IN THE OTHER WORLD EVERYTHING IS HUGE. IN THIS ONE, IT'S YOU."],
           style=2, yaw=math.pi)
    ab.enemy('blob', 3, g(3, 5) + 1, 5, 3)
    ab.coins_ring(0, g(0, 0) + 1, 0, 3, 6)


def build_giant(ab):
    ab.fog = (235, 220, 190)
    ab.sky = (ab.fog, ab.fog, ab.fog)
    ab.view = 200
    wall = (230, 205, 165)
    make_room(ab, -60, -60, 60, 60, 0, 70, wall, (150, 100, 60), (240, 230, 210), checker=(170, 118, 70),
              gaps={'s': [(0, 3.0, 4.4)], 'w': [(20, 1.4, 2.0)]}, tile=8.0)
    wood = (160, 110, 60)
    for (lx, lz) in ((6, 10), (34, 10), (6, 30), (34, 30)):
        ab.prism(lx, lz, 0, 18, 1.3, 8, cmul(wood, 0.9))
    ab.box(3, 18, 7, 37, 19.5, 33, wood, tile=8)
    for i in range(9):
        c = hsv(0.08 * i, 0.55, 0.85)
        ab.box(-12 + i * 1.6, 2.0 * i, 22, 3.6, 2.0 * (i + 1), 30, c)
    ab.portal('twin', 20, 19.5 + 1.2, 20, 0.0, 'pipe', ('tiny', 0, 'twin'))
    ab.star(30, 20.6, 26)
    for i in range(10):
        x = -38 + (i % 2) * 2.2
        z = -30 + (i % 3) * 1.8
        ab.box(x - 2, 1.9 * i, z - 2, x + 2, 1.9 * (i + 1), z + 2, hsv(i * 0.13, 0.7, 0.95))
    ab.star(-37, 1.9 * 10 + 0.9, -29)
    ab.prism(-20, 30, 0, 12, 6, 12, (220, 60, 60), colr=[(220, 60, 60), (240, 240, 240)])
    ab.box(-50, 0, -10, -40, 10, 10, (120, 80, 50))
    ab.box(-50, 10, -10, -48, 26, 10, (120, 80, 50))
    ab.portal('entry', 0, 0, -60, 0.0, 'door', ('castle', 1, 'p_snow'), w=3.0, h=4.4)
    ab.portal('hole', -60, 0, 20, HALF_PI, 'door', ('backrooms', 0, 'entry'), w=1.4, h=2.0, color=(40, 30, 30))
    ab.enemy('blob', 10, 0.5, -20, 8)
    ab.enemy('blob', -20, 0.5, 0, 8)
    ab.npc(5, 0, -45, "GOOMBA", ["EVERYTHING IS SO BIG. OR WE ARE SO SMALL. NOBODY WILL TELL ME WHICH.",
                                     "THE BOOKS BY THE TABLE ARE STAIRS IF YOU'RE BRAVE.",
                                     "THE MOUSE HOLE IN THE WEST WALL SMELLS LIKE OLD CARPET AND HUMMING."],
           style=5, yaw=0.0)
    ab.coins_line((0, 0.8, -50), (0, 0.8, -20), 6)
    ab.spawn('entry', 0, 0, -56, 0.0)


def build_maze(ab):
    ab.fog = (70, 76, 72)
    ab.sky = ((40, 42, 45), (90, 95, 90), (40, 42, 45))
    ab.amb = 0.52
    ab.view = 100
    doors = {('s', 4): 'entry', ('w', 6): 'lava', ('e', 2): 'basement', ('n', 2): 'tomb'}
    cc, dist, pos = build_grid_maze(ab, 9, 8.0, 6.0, (130, 125, 120), (90, 90, 95), doors, extra=0.12)
    rules = {'entry': ('castle', 1, 'p_lava'), 'lava': ('lava', 0, 'vent'), 'basement': ('basement', 4, 'side'),
             'tomb': ('desert', 0, 'tomb')}
    for name, (x, z, yaw) in pos.items():
        ab.portal(name, x, 0, z, yaw, 'door', rules[name], color=(200, 120, 40))
    cx, cz = cc(4, 4)
    ab.portal('core', cx, 1.2, cz, 0.0, 'pipe', ('city', 0, 'maze'))
    far = sorted(dist.items(), key=lambda kv: -kv[1])
    used = {(4, 4)}
    k = 0
    for (cell, dd) in far:
        if cell in used:
            continue
        x, z = cc(*cell)
        if k == 0:
            ab.star(x, 1.2, z)
        else:
            ab.box(x - 1.2, 0, z - 1.2, x + 1.2, 2.2, z + 1.2, (160, 110, 60))
            ab.star(x, 3.2, z)
        used.add(cell)
        k += 1
        if k == 2:
            break
    rng = ab.rng
    for i in range(10):
        x0 = -36 + rng.randrange(9) * 8
        ab.box(x0, 6, -36 + rng.randrange(9) * 8 - 0.3, x0 + 8, 6.5, -36 + rng.randrange(9) * 8 + 0.3,
               (160, 90, 40), solid=False)
    cells = [c for c in dist if c not in used and c != (4, 0)]
    rng.shuffle(cells)
    for i, c in enumerate(cells[:5]):
        x, z = cc(*c)
        ab.enemy('cube' if i % 2 else 'blob', x, 0.6, z, 3.0, axis=i % 2)
    x, z = cc(4, 0)
    ab.npc(x + 2.5, 0, z, "BOB-OMB FOREMAN", ["THE FACTORY MAKES CORRIDORS. WE'RE AHEAD OF SCHEDULE.",
                                      "THE PIPE IN THE MIDDLE DRAINS INTO A CITY. DON'T TELL THE CITY."],
           style=4, yaw=-HALF_PI)
    ab.coins_line((x, 0.8, z - 2), (x, 0.8, z + 10), 4)


def build_stairs(ab):
    ab.fog = (60, 40, 92)
    ab.sky = ((20, 10, 40), (80, 50, 120), (20, 10, 40))
    ab.hue_cycle = True
    ab.view = 120
    R = 15.0
    ab.prism(0, 0, 0, 60, R, 16, (90, 70, 130), inward=True, cap=False, colr=[(90, 70, 130), (80, 60, 120)])
    ab.prism(0, 0, -1, 0, R + 0.5, 16, (70, 50, 100), top=(120, 100, 150))
    ab.prism(0, 0, 0, 43.4, 2.5, 10, (150, 130, 190))
    a0 = -HALF_PI + 0.35
    n = 100
    for i in range(n):
        a = a0 + i * 0.17
        px, pz = math.cos(a) * 8.5, math.sin(a) * 8.5
        top = 0.44 * (i + 1)
        s = 2.2 if i % 25 == 24 else 1.3
        c = (200, 180, 230) if i % 2 else (170, 150, 210)
        ab.box(px - s, top - 0.6, pz - s, px + s, top, pz + s, c)
        if i == 50:
            ex, ez = math.cos(a) * 12.6, math.sin(a) * 12.6
            ab.box(ex - 1, top + 0.4, ez - 1, ex + 1, top + 1.0, ez + 1, (230, 210, 255))
            ab.switch(ex, top + 1.0, ez, 'stairs_sw', "THE TOP OF THE STAIRS STOPS PRETENDING.")
    ab.prism(0, 0, 43.4, 44.0, 6.0, 8, (210, 190, 240))
    ab.star(0, 46.3, 0)
    ab.portal('top', 0, 44.0, -3.5, 0.0, 'door',
              [(lambda w: len(w.stars) >= 20 or 'stairs_sw' in w.flags, ('observatory', 0, 'stairs')),
               (None, ('stairs', 0, 'entry'))], free=True, color=(240, 220, 120))
    ab.portal('entry', 0, 0, -R + 1.2, 0.0, 'door', ('castle', 0, 'star_door'), free=True, color=(240, 220, 120))
    ab.npc(-4, 0, -10, "ENDLESS TOAD", ["ONE HUNDRED STEPS. I'VE COUNTED THEM A THOUSAND TIMES. SOMETIMES IT'S 101.",
                                         "THE DOOR AT THE TOP IS POLITE. IT SENDS YOU BACK UNTIL YOU'RE READY.",
                                         "TWENTY POWER STARS, OR A SWITCH HANGING OFF THE MIDDLE OF THE CLIMB."],
           style=2, yaw=0.3)
    for i in range(5, n, 10):
        a = a0 + i * 0.17
        ab.coin(math.cos(a) * 8.5, 0.44 * (i + 1) + 0.8, math.sin(a) * 8.5)
    ab.spawn('entry', 0, 0, -5.0, 0.0)


def build_backrooms(ab):
    ab.fog = (180, 170, 100)
    ab.sky = (ab.fog, ab.fog, ab.fog)
    ab.amb = 0.78
    ab.view = 70
    ab.fog_near = 0.12
    doors = {('s', 5): 'entry', ('w', 3): 'door_a', ('e', 7): 'door_b', ('n', 2): 'door_c', ('n', 8): 'door_d'}
    cc, dist, pos = build_grid_maze(ab, 11, 6.0, 3.2, (205, 190, 110), (150, 130, 80), doors, extra=0.35,
                                    ceil=(215, 205, 150), t=0.5)
    rules = {'entry': ('castle', 0, 'left'), 'door_a': ('lobby', 0, 'd2'), 'door_b': ('castle', 1, 'left'),
             'door_c': ('backrooms', 0, 'door_d'), 'door_d': ('giant', 0, 'hole')}
    for name, (x, z, yaw) in pos.items():
        ab.portal(name, x, 0, z, yaw, 'door', rules[name], color=(170, 150, 90))
    for i in range(11):
        for j in range(11):
            if (i + j) % 2 == 0:
                x, z = cc(i, j)
                ab.box(x - 1, 3.12, z - 0.4, x + 1, 3.19, z + 0.4, (255, 255, 230), solid=False, raw=True, bias=0.3)
    far = sorted(dist.items(), key=lambda kv: -kv[1])
    for (cell, dd) in far[:2]:
        x, z = cc(*cell)
        ab.star(x, 1.2, z)
    rng = ab.rng
    cells = list(dist.keys())
    c = cells[rng.randrange(len(cells))]
    x, z = cc(*c)
    ab.npc(x, 0, z, "BIG BOO", ["BZZZZZZZZZZZZZZZZZ.", "THE LIGHTS ARE THE ONLY THING HERE THAT KNOWS THE WAY.",
                                "ONE DOOR LEADS TO ANOTHER DOOR IN THIS SAME PLACE. POLITE, ISN'T IT?"], style=3)
    c2 = cells[rng.randrange(len(cells))]
    x, z = cc(*c2)
    ab.enemy('spike', x, 1.5, z, 3)


def build_garden(ab):
    deeper = ab.v == 1
    seed = ab.w.seed + (77 if deeper else 0)
    rng = ab.rng
    ab.wobble = True
    ab.hue_cycle = True
    ab.glitch = True
    ab.sky = ((255, 120, 200), (120, 255, 220), (60, 20, 80)) if not deeper else ((20, 255, 120), (255, 40, 200),
                                                                                    (0, 0, 0))
    ab.fog = (200, 160, 255) if not deeper else (120, 255, 200)

    def H(x, z):
        h = fbm(x * 0.04, z * 0.04, seed, 3) * 8 - 3 + math.sin(x * 0.2) * math.cos(z * 0.2) * (2 if deeper else 0.8)
        r = max(abs(x), abs(z))
        if r > 52:
            h += (r - 52) * 0.8
        return h

    def cfn(x, z, h, i, j):
        r = _h2(i, j, seed)
        if r < (0.22 if deeper else 0.1):
            return hsv(_h2(j, i, seed + 1), 1.0, 1.0)
        return lerpc((80, 200, 110), (230, 120, 200), vnoise(x * 0.1, z * 0.1, seed))

    ab.heightfield(-60, -60, 5, 24, 24, H, cfn)
    for (a, b, c, d) in ((-61, -61, -60, 61), (60, -61, 61, 61), (-61, -61, 61, -60), (-61, 60, 61, 61)):
        ab.box(a, -10, b, c, 50, d, (0, 0, 0), visible=False)
    x, y, z = 10.0, H(10, 10), 10.0
    for i in range(9):
        x += rng.uniform(-3, 3)
        z += rng.uniform(2.5, 3.5)
        y += 1.7
        s = rng.uniform(0.9, 1.4)
        ab.box(x - s, y - s, z - s, x + s, y, z + s, hsv(rng.random(), 0.9, 1.0), raw=True)
    ab.portal('glitch', x, y + 2.4, z + 0.5, 0.0, 'ring',
              [(need(30), ('dream', 0, 'glitch')), (None, ('garden', 0, 'glitch'))] if deeper else
              ('garden', 1, 'glitch'), fy=y)
    ab.star(x + 1.5, y + 1.0, z - 1)
    for i in range(12):
        tx, tz = rng.uniform(-50, 50), rng.uniform(-50, 50)
        ty = H(tx, tz) + rng.uniform(6, 14)
        ab.prism(tx, tz, ty, ty + 3, 0.0, 6, hsv(rng.random(), 0.8, 1.0), rt=2.2, solid=False, raw=True, cap=True)
        ab.prism(tx, tz, ty + 3, ty + 5, 0.3, 5, (120, 80, 60), solid=False)
    for i in range(10):
        tx, tz = rng.uniform(-50, 50), rng.uniform(-50, 50)
        make_tree(ab, tx, tz, H(tx, tz), rng.uniform(1, 1.6), leaf=hsv(rng.random(), 0.7, 0.9))
    for i in range(18):
        bx, bz = rng.uniform(-55, 55), rng.uniform(-55, 55)
        by = H(bx, bz) + rng.uniform(3, 20)
        s = rng.uniform(0.4, 1.6)
        ab.box(bx - s, by - s, bz - s, bx + s, by + s, bz + s, hsv(rng.random(), 1.0, 1.0), solid=False, raw=True)
    if not deeper:
        ix, iz = -30, 25
        iy = H(ix, iz) + 7
        make_island(ab, ix, iy, iz, 3.5, top=(255, 80, 200), rock=(120, 255, 240))
        for k in range(3):
            ab.box(ix + 6 + k * 3 - 1, iy - 6 + k * 1.9 - 0.4, iz - 1, ix + 6 + k * 3 + 1, iy - 6 + k * 1.9, iz + 1,
                   hsv(k * 0.3, 1, 1), raw=True)
        ab.star(ix, iy + 1.2, iz)
    g0 = H(0, -50)
    g0 = H(0, -57)
    ab.portal('entry', 0, g0 + 2.2, -57, 0.0, 'ring', ('grounds', 1 if deeper else 0, 'hollow'), fy=g0)
    ab.spawn('entry', 0, H(0, -47) + 0.2, -47, 0.0)
    gg = H(-30, 0)
    ab.portal('gate', -30, gg, 0, HALF_PI, 'door', ('mirror', 0, 'front') if deeper else ('courtyard', 0, 'gate'),
              free=True, color=(255, 0, 255))
    ab.enemy('spike', 0, H(0, 0) + 2, 0, 8, col=(0, 255, 180))
    ab.enemy('spike', 20, H(20, -20) + 2, -20, 8, col=(255, 60, 60))
    ab.enemy('blob', -15, H(-15, -25) + 1, -25, 6, col=(255, 0, 255))
    ab.npc(6, H(6, -44), -44, "METAL TOAD" if not deeper else "M3TAL T0AD",
           ["HELLO HELLO HELLO. I WAS RENDERED WRONG AND I LIKE IT.",
            "THE BLOCKS GO UP TO A RING. THE RING GOES DEEPER. DEEPER GOES DEEPER.",
            "IF YOU CARRY THIRTY POWER STARS THE GLITCH BECOMES A DOOR TO THE LAST COURSE."], style=6, yaw=0.0)
    ab.coins_ring(0, H(0, -35) + 1, -35, 4, 7)


def build_lobby(ab):
    again = ab.v == 1
    wall = (230, 190, 200) if again else (190, 200, 190)
    floor, floor2 = ((220, 120, 160), (30, 20, 30)) if again else ((60, 150, 150), (230, 225, 200))
    ab.fog = (200, 150, 170) if again else (150, 160, 150)
    ab.sky = (ab.fog, ab.fog, ab.fog)
    ab.amb = 0.62
    ab.view = 110
    gaps = {'s': [(0, 2.2, 3.4)], 'w': [(8, 2.2, 3.4), (20, 2.2, 3.4), (32, 2.2, 3.4)],
            'e': [(8, 2.2, 3.4), (20, 2.2, 3.4), (32, 2.2, 3.4)], 'n': [(0, 2.2, 3.4)]}
    if again:
        gaps['n'].append((16, 2.4, 3.0, 'fake', 'lobby_again_alcove'))
    make_room(ab, -24, 0, 24, 40, 0, 9, wall, floor, cmul(wall, 0.8), checker=floor2, gaps=gaps)
    for (x, z) in ((-12, 12), (12, 12), (-12, 28), (12, 28)):
        ab.prism(x, z, 0, 9, 1.1, 8, cmul(wall, 0.9))
    ab.box(-5, 0, 30, 5, 1.2, 32, (120, 80, 60))
    for (x, z) in ((-20, 4), (20, 4), (-20, 36), (20, 36)):
        ab.prism(x, z, 0, 1.0, 0.8, 6, (150, 90, 60))
        ab.prism(x, z, 1.0, 3.2, 1.3, 6, (60, 150, 70), rt=0.0, solid=False)
    ab.box(-8, 0, 16, -4, 0.8, 18, (110, 90, 70))
    ab.box(4, 0, 16, 8, 0.8, 18, (110, 90, 70))
    if again:
        ab.box(14, -1, 40, 18, 0, 44, floor)
        wall_z(ab, 40, 44, 14, 0, 4, wall)
        wall_z(ab, 40, 44, 18, 0, 4, wall)
        wall_x(ab, 14, 18, 44, 0, 4, wall)
        ab.box(13.7, 4, 40, 18.3, 4.5, 44.3, wall)
        ab.star(16, 1.2, 42.5)
        R = {'entry': ('lobby', 0, 'd6'), 'd1': ('mirror', 0, 'front'), 'd2': ('dark', 0, 'front'),
             'd3': ('tiny', 0, 'entry'), 'd4': ('maze', 0, 'entry'), 'd5': ('city', 0, 'lobby'),
             'd6': ('lobby', 0, 'entry'), 'd7': ('stairs', 0, 'entry')}
    else:
        for k, (y, z) in enumerate(((2.4, 27.0), (4.2, 24.0), (6.0, 21.0))):
            ab.box(-1, y - 0.4, z - 1, 1, y, z + 1, (0, 0, 0), visible=False)
        ab.trigger(-1, 2.3, 26, 1, 3.5, 28, 'secret', 'lobby_invisible', "YOU'RE STANDING ON NOTHING.")
        ab.star(0, 7.3, 18.5)
        R = {'entry': ('castle', 0, 'left'), 'd1': ('castle', 1, 'left'), 'd2': ('backrooms', 0, 'door_a'),
             'd3': ('haunted', 1, 'library'), 'd4': ('giant', 0, 'entry'), 'd5': ('observatory', 0, 'lobby'),
             'd6': ('lobby', 1, 'entry'), 'd7': ('grounds', 1, 'front')}
    fc = (200, 190, 120)
    ab.portal('entry', 0, 0, 0, 0.0, 'door', R['entry'], color=fc)
    for k, z in enumerate((8, 20, 32)):
        ab.portal('d%d' % (k + 1), -24, 0, z, HALF_PI, 'door', R['d%d' % (k + 1)], color=hsv(k * 0.15, 0.5, 0.9))
        ab.portal('d%d' % (k + 4), 24, 0, z, -HALF_PI, 'door', R['d%d' % (k + 4)],
                  lock=(need(15), "THIS DOOR WANTS 15 POWER STARS.") if (k == 1 and not again) else None,
                  color=hsv(0.5 + k * 0.15, 0.5, 0.9))
    ab.portal('d7', 0, 0, 40, math.pi, 'door', R['d7'], color=(240, 240, 240))
    ab.npc(0, 0, 34, "LOBBY TOAD" if not again else "LOBBY TOAD (AGAIN)",
           ["WELCOME TO THE LOBBY. PLEASE TAKE A DOOR. ANY DOOR. THEY ALL CHECK OUT EVENTUALLY.",
            "THE CEILING ABOVE THE DESK IS CLOSER THAN IT LOOKS. SO IS THE AIR." if not again else
            "YOU'VE BEEN HERE BEFORE. THE FLOOR WAS A DIFFERENT COLOUR THEN. SO WERE YOU.",
            "DOOR SIX GOES TO A LOBBY. THIS LOBBY. OR THE OTHER ONE."], style=2, yaw=math.pi)
    ab.coins_line((-6, 0.8, 7), (6, 0.8, 7), 5)


def build_observatory(ab):
    ab.starfield = True
    ab.sky = ((5, 5, 25), (40, 30, 90), (10, 10, 30))
    ab.fog = (30, 25, 72)
    ab.amb = 0.55
    ab.view = 160
    R = 24.0
    ab.prism(0, 0, 0, 14, R, 16, (70, 70, 110), inward=True, cap=False, colr=[(70, 70, 110), (60, 60, 100)])
    ab.prism(0, 0, -1, 0, R + 0.5, 16, (50, 50, 80), top=(90, 90, 130))
    ab.prism(0, 0, 0, 3, 3.2, 10, (120, 110, 140))
    for i in range(6):
        ab.prism(1.2 * (i + 1), 0, 3 + i * 2, 5 + i * 2, 1.8, 10, (170, 170, 200) if i % 2 else (140, 140, 180))
    ab.star(7.2, 15.9, 0)
    ab.portal('telescope', 8.4, 17.8, 0, HALF_PI, 'ring', ('dream', 0, 'entry'),
              lock=(need(30), "THE TELESCOPE FOCUSES ONLY FOR 30 POWER STARS."), fy=15)
    ab.mover(0, 4.0, 0, 1.6, 0.3, 1.6, 'circle', (14.0, 0, 0), 0.35, 0.0, (200, 200, 255))
    ab.mover(0, 8.0, 0, 1.6, 0.3, 1.6, 'circle', (14.0, 0, 0), -0.3, 2.0, (200, 200, 255))
    ab.box(-19.5, 9.5, -1.5, -16.5, 10.0, 1.5, (220, 220, 255))
    ab.star(-18, 11.0, 0)
    ab.box(3.3, 0, -1, 5.3, 1.6, 1, (100, 100, 140))
    for i, (x, y, z, r, c) in enumerate(((0, 24, 0, 3, (255, 200, 90)), (-12, 20, 10, 1.5, (120, 180, 255)),
                                         (10, 22, -12, 2, (255, 120, 120)))):
        ab.prism(x, z, y - r, y + r, r, 8, c, rt=r * 0.4, solid=False, raw=True)
    fc = (170, 170, 230)
    ab.portal('entry', 0, 0, -R + 1.3, 0.0, 'door', ('castle', 0, 'right'), free=True, color=fc)
    ab.portal('stairs', R - 1.3, 0, 0, -HALF_PI, 'door', ('stairs', 0, 'top'), free=True, color=fc)
    ab.portal('clock', -R + 1.3, 0, 0, HALF_PI, 'door', ('clock', 0, 'face'), free=True, color=fc)
    ab.portal('lobby', 0, 0, R - 1.3, math.pi, 'door', ('lobby', 0, 'd5'), free=True, color=fc)
    ab.npc(-5, 0, -14, "STAR LAKITU", ["EVERY STAR UP THERE IS A POWER STAR SOMEONE DIDN'T PICK UP.",
                                      "THE TELESCOPE SEES THE LAST COURSE WHEN YOU CARRY THIRTY.",
                                      "RIDE THE ORBITS. THE HIGH LEDGE WEST HOLDS SOMETHING."], style=4, yaw=0.2)
    ab.coins_ring(0, 0.8, 0, 10, 12)
    ab.spawn('entry', 0, 0, -14.0, 0.0)


def build_city(ab):
    rng = ab.rng
    ab.sky = ((5, 5, 12), (35, 28, 48), (8, 6, 14))
    ab.fog = (36, 30, 48)
    ab.amb = 0.5
    ab.view = 120
    ab.box(-62, -1, -62, 62, 0, 62, (70, 70, 80), checker=(80, 80, 92), tile=12)
    ab.qgrid((-62, 45, -62), (62, 45, -62), (62, 45, 62), (-62, 45, 62), 8, 8, (40, 34, 45), (0, -1, 0))
    ab.col_box(-62, 45, -62, 62, 50, 62)
    for (a, b, c, d) in ((-63, -63, -62, 63), (62, -63, 63, 63), (-63, -63, 63, -62), (-63, 62, 63, 63)):
        ab.box(a, -2, b, c, 45, d, (60, 50, 60), tile=12)
    heights = {}
    chain = [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]
    for bi in range(5):
        for bj in range(5):
            if (bi, bj) == (2, 2):
                continue
            cx, cz = -48 + bi * 24, -48 + bj * 24
            if (bi, bj) in chain:
                h = 6 + chain.index((bi, bj)) * 4
            else:
                h = rng.choice((8, 12, 16, 20, 26))
            heights[(bi, bj)] = h
            col = hsv(0.6 + rng.uniform(-0.1, 0.1), 0.25, rng.uniform(0.35, 0.55))
            if (bi, bj) == (3, 3):
                make_room(ab, cx - 7, cz - 7, cx + 7, cz + 7, 0, h, col, (60, 60, 70), cmul(col, 0.8),
                          gaps={'s': [(cx, 2.4, 3.0, 'fake', 'city_hollow')]})
                ab.star(cx, 1.2, cz)
            else:
                ab.box(cx - 7, 0, cz - 7, cx + 7, h, cz + 7, col, top=cmul(col, 0.8), tile=7)
            for k in range(int(h // 4)):
                y = 2 + k * 4
                if rng.random() < 0.8:
                    ab.box(cx - 5, y, cz - 7.05, cx - 3, y + 1.5, cz - 6.95, (255, 220, 120), solid=False, raw=True,
                           bias=0.3)
                if rng.random() < 0.8:
                    ab.box(cx + 3, y, cz - 7.05, cx + 5, y + 1.5, cz - 6.95, (255, 200, 90), solid=False, raw=True,
                           bias=0.3)
    for k in range(4):
        a, b = chain[k], chain[k + 1]
        ha, hb = heights[a], heights[b]
        xa = -48 + a[0] * 24 + 7
        xb = -48 + b[0] * 24 - 7
        cz = -48 + a[1] * 24
        ab.ramp(xa, cz - 1.5, xb, cz + 1.5, max(0, min(ha, hb) - 1.0), ha, hb, 0, (150, 110, 70))
    x0 = -48 - 7
    make_stairs(ab, x0 - 1.5, -24 - 7 + 0.5, 0, 15, 0, 1, 3, 0.4, 0.9, (120, 120, 130))
    ab.star(48, heights[(4, 1)] + 1.2, -24)
    ab.prism(0, 0, 0, 3, 2.5, 10, (120, 120, 140), top=(60, 100, 160))
    ab.portal('well', 0, 8.8, 0, 0.0, 'ring', ('grounds', 1, 'well'), fy=3)
    for (lx, lz) in ((9, 9), (-9, 9), (9, -9), (-9, -9)):
        ab.prism(lx, lz, 0, 5, 0.2, 5, (40, 40, 50))
        ab.prism(lx, lz, 5, 5.6, 0.8, 6, (255, 240, 180), raw=True)
    ab.box(7.6, 0, 11.5, 9.8, 2.0, 13.5, (110, 90, 70))
    ab.box(7.8, 0, 10.2, 9.6, 3.8, 11.4, (100, 80, 60))
    ab.box(-1, 0, -4.2, 1, 1.4, -3, (110, 90, 70))
    ab.star(9, 6.5, 9)
    fc = (150, 140, 180)
    ab.portal('stairs', -6, 0, -8, 0.0, 'door', ('basement', 9, 'down'), free=True, color=fc)
    ab.portal('caves', 6, 0, -8, 0.0, 'door', ('caverns', 0, 'deep'), free=True, color=fc)
    ab.portal('lobby', -6, 0, 8, math.pi, 'door', ('lobby', 1, 'd5'), free=True, color=fc)
    ab.portal('pyramid', 6, 2.3, 8, math.pi, 'ring', ('desert', 0, 'pyramid'), fy=0)
    ab.portal('maze', -10, 1.2, 0, 0.0, 'pipe', ('maze', 0, 'core'))
    for (x, z) in ((0, -20), (20, 0), (-20, 20)):
        ab.enemy('blob', x, 0.5, z, 5, col=(150, 100, 200))
    ab.npc(4, 0, -4, "CITY TOAD", ["NOBODY HERE HAS SEEN THE SUN. WE HEARD IT'S LIKE A LAMP BUT RUDER.",
                                     "THE ROOFS NORTH OF HERE ARE LINKED BY PLANKS. THE TALLEST HOLDS A POWER STAR.",
                                     "ONE BUILDING TO THE SOUTH-EAST IS EMPTY INSIDE. ITS FRONT DOOR IS A RUMOUR."],
           style=4, yaw=math.pi)
    ab.npc(-4, 0, 4, "TOAD CLERK", ["THE RING ABOVE THE FOUNTAIN GOES UP A WELL. JUMP THREE TIMES.",
                               "THIS CITY IS BELOW EVERY OTHER PLACE. ALL THE DRAINS END HERE."], style=1, yaw=0.0)
    ab.coins_ring(0, 0.8, 0, 12, 12)
    ab.spawn('entry', 0, 0, -12, 0.0)


def build_dream(ab):
    rng = ab.rng
    ab.hue_cycle = True
    ab.starfield = True
    ab.wobble = True
    ab.sky = ((255, 190, 240), (200, 240, 255), (255, 250, 220))
    ab.fog = (255, 228, 245)
    ab.kill_y = -50
    ab.view = 200
    ab.box(-5, -1, -67, 5, 0, -55, (255, 255, 255), checker=(255, 210, 240))
    ab.portal('entry', -2, 2.3, -65, 0.0, 'ring', ('castle', 0, 'front'), fy=0)
    ab.portal('glitch', 2.5, 2.3, -65, 0.0, 'ring', ('garden', 1, 'glitch'), fy=0)
    ab.spawn('entry', 0, 0, -56.5, 0.0)
    z = -56.0
    x = 0.0
    y = 0.0
    for i in range(12):
        z += rng.uniform(3.8, 4.8)
        x = clamp(x + rng.uniform(-3, 3), -8, 8)
        y += rng.choice((0.0, 0.6, 1.0))
        c = hsv(i / 12.0 + 0.5, 0.35, 1.0)
        if i % 4 == 3:
            ab.mover(x, y, z, 1.4, 0.3, 1.4, 'line', (0, 1.2, 0), 1.0, i, c)
        elif i % 3 == 0:
            ab.prism(x, z, y - 0.8, y, 1.7, 6, c, raw=True, rot=i)
        else:
            ab.box(x - 1.5, y - 0.6, z - 1.5, x + 1.5, y, z + 1.5, c, raw=True)
    ab.prism(0, 0, y - 1.5, y, 12, 12, (255, 255, 255), top=(255, 240, 250), colr=[(255, 220, 240), (220, 240, 255)])
    ab.star(0, y + 2.4, 0, final=True, label="THE LAST POWER STAR")
    for i in range(8):
        a = TAU * i / 8
        ab.prism(math.cos(a) * 9, math.sin(a) * 9, y, y + 4 + (i % 3), 0.6, 6, hsv(i / 8, 0.4, 1.0), raw=True)
    for i in range(20):
        px, py, pz = rng.uniform(-60, 60), rng.uniform(-20, 40), rng.uniform(-60, 60)
        if math.hypot(px, pz) < 16:
            continue
        s = rng.uniform(1.5, 5)
        if i % 2:
            ab.box(px - s, py - s, pz - s, px + s, py + s, pz + s, hsv(rng.random(), 0.3, 1.0), solid=False, raw=True)
        else:
            ab.prism(px, pz, py - s, py + s, s, 5, hsv(rng.random(), 0.3, 1.0), rt=0.0, solid=False, raw=True)
    ab.npc(-4, y, -6, "SHADOW MARIO", ["YOU CAME ALL THIS WAY TO FIND THE END OF THE COURSE. IT'S RIGHT THERE.",
                                    "WHEN YOU WAKE UP THE CASTLE WILL STILL BE HERE. IT ALWAYS IS."],
           style=6, yaw=0.5)


def build_rare(ab, kind):
    if kind == 'null':
        ab.sky = ((0, 0, 0), (0, 0, 0), (0, 0, 0))
        ab.fog = (0, 0, 0)
        ab.amb = 0.9
        ab.box(-10, -1, -10, 10, 0, 10, (240, 240, 240), checker=(200, 200, 200))
        ab.star(0, 1.2, 6)
        ab.npc(-4, 0, 2, "VANISH BOO", ["...", "YOU WEREN'T SUPPOSED TO COUNT THE DOOR THAT MANY TIMES.",
                                 "THIS ROOM HAS NO NAME. PLEASE DON'T GIVE IT ONE."], style=6)
        ab.portal('exit', 0, 0, -8, 0.0, 'door', ('lobby', 0, 'entry'), free=True, color=(255, 255, 255))
        ab.spawn('entry', 0, 0, -5, 0.0)
        ab.kill_y = -30
    elif kind == 'parlor':
        ab.fog = (120, 80, 50)
        ab.sky = (ab.fog, ab.fog, ab.fog)
        make_room(ab, -7, -7, 7, 7, 0, 5, (170, 110, 80), (120, 60, 40), (90, 50, 30), checker=(140, 70, 45),
                  gaps={'s': [(0, 2.2, 3.4)]})
        ab.box(-3, 0, 2, 3, 0.9, 4, (90, 50, 30))
        ab.prism(-5, 5, 0, 1.2, 0.8, 8, (200, 60, 60))
        ab.prism(5, 5, 0, 1.2, 0.8, 8, (200, 60, 60))
        ab.star(0, 1.9, 3)
        ab.npc(4, 0, -2, "PEACH", ["OH, A GUEST. NOBODY FINDS THE PARLOR ON PURPOSE.",
                                      "TEA? NO? THE CASTLE DOESN'T DRINK EITHER.",
                                      "TAKE THE POWER STAR. IT'S BEEN WAITING LONGER THAN I HAVE."], style=2,
               yaw=math.pi)
        ab.portal('exit', 0, 0, -7, 0.0, 'door', ('lobby', 0, 'entry'), color=(200, 160, 100))
        ab.spawn('entry', 0, 0, -4, 0.0)
    else:
        ab.fog = (128, 128, 128)
        ab.sky = ((40, 40, 40), (128, 128, 128), (60, 60, 60))
        ab.amb = 0.65
        make_room(ab, -5, 0, 5, 90, 0, 7, (150, 150, 150), (90, 90, 90), (60, 60, 60), checker=(170, 170, 170),
                  gaps={'s': [(0, 2.2, 3.4)]})
        for z in range(8, 88, 8):
            for sx in (-4, 4):
                ab.prism(sx, z, 0, 7, 0.5, 6, (200, 200, 200))
        ab.star(0, 1.2, 86)
        ab.npc(0, 0, 45, "DRY BONES", ["COLOUR IS A RUMOUR.", "THE POWER STAR AT THE END IS THE ONLY BRIGHT THING HERE."],
               style=3, yaw=math.pi)
        ab.portal('exit', 0, 0, 0, 0.0, 'door', ('lobby', 0, 'entry'), color=(210, 210, 210))
        ab.spawn('entry', 0, 0, 3, 0.0)


# ----------------------------------------------------------------------------
# /* levels/level_defines.h */  area registry (B3313 1.0)
# ----------------------------------------------------------------------------
AREAS = {
    'grounds': ("CASTLE GROUNDS", 2, 'calm', lambda ab: build_grounds(ab)),
    'castle': ("CASTLE LOBBY", 2, 'calm', lambda ab: build_castle(ab, 'alt' if ab.v == 1 else 'main')),
    'basement': ("BASEMENT", 10, 'eerie', build_basement),
    'courtyard': ("CASTLE COURTYARD", 1, 'calm', build_courtyard),
    'snow': ("COOL COOL MOUNTAIN", 1, 'calm', build_snow),
    'lava': ("LETHAL LAVA LAND", 1, 'eerie', build_lava),
    'desert': ("SHIFTING SAND LAND", 1, 'calm', build_desert),
    'haunted': ("BIG BOO'S HAUNT", 2, 'eerie', build_haunted),
    'clock': ("TICK TOCK CLOCK", 1, 'dream', build_clock),
    'sky': ("WING MARIO OVER THE RAINBOW", 1, 'dream', build_sky),
    'caverns': ("HAZY MAZE CAVE", 1, 'eerie', build_caverns),
    'rainbow': ("RAINBOW RIDE", 1, 'dream', build_rainbow),
    'aquarium': ("DIRE DIRE DOCKS", 1, 'eerie', build_aquarium),
    'dark': ("DARK LOBBY", 1, 'eerie', lambda ab: build_castle(ab, 'dark')),
    'mirror': ("MIRROR LOBBY", 1, 'dream', lambda ab: build_castle(ab, 'mirror')),
    'tiny': ("TINY-HUGE ISLAND (TINY)", 1, 'calm', build_tiny),
    'giant': ("TINY-HUGE ISLAND (HUGE)", 1, 'calm', build_giant),
    'maze': ("WHOMP'S FORTRESS", 1, 'eerie', build_maze),
    'stairs': ("ENDLESS STAIRS", 1, 'dream', build_stairs),
    'backrooms': ("CREEPY CORRIDORS", 1, 'eerie', build_backrooms),
    'garden': ("BOB-OMB BATTLEFIELD", 2, 'dream', build_garden),
    'lobby': ("PLEXAL LOBBY", 2, 'eerie', build_lobby),
    'observatory': ("TOWER OF THE WING CAP", 1, 'dream', build_observatory),
    'city': ("CAVERN OF THE METAL CAP", 1, 'eerie', build_city),
    'dream': ("BOWSER IN THE DREAM", 1, 'dream', build_dream),
    'null': ("THE VOID", 1, 'eerie', lambda ab: build_rare(ab, 'null')),
    'parlor': ("PEACH'S SECRET SLIDE", 1, 'calm', lambda ab: build_rare(ab, 'parlor')),
    'mono': ("MONOCHROME LOBBY", 1, 'eerie', lambda ab: build_rare(ab, 'mono')),
}
RARE_AREAS = ('null', 'parlor', 'mono')
VARIANT_NAMES = {('grounds', 1): "CASTLE GROUNDS (NIGHT)", ('castle', 1): "PARALLEL LOBBY",
                 ('haunted', 1): "BIG BOO'S HAUNT (OTHER)", ('garden', 1): "BOB-OMB BATTLEFIELD (DEEPER)",
                 ('lobby', 1): "PLEXAL LOBBY (AGAIN)"}
RARE_HOST_CANDIDATES = [('castle', 0, 'left'), ('castle', 0, 'right'), ('lobby', 0, 'd6'), ('lobby', 1, 'd3'),
                        ('backrooms', 0, 'door_c'), ('haunted', 0, 'cellar'), ('basement', 5, 'down'),
                        ('stairs', 0, 'top'), ('city', 0, 'lobby'), ('courtyard', 0, 'entry'),
                        ('snow', 0, 'cave'), ('grounds', 0, 'pipe')]


def area_name(aid, v):
    if aid == 'basement':
        return "BASEMENT FLOOR %d" % (v + 1)
    return VARIANT_NAMES.get((aid, v), AREAS[aid][0])


def node_key(aid, v):
    if aid == 'basement' or v == 0:
        return aid
    return "%s#%d" % (aid, v)


def build_area(world, aid, v, dry=False):
    name, nv, music, fn = AREAS[aid]
    v = clamp(v, 0, nv - 1)
    ab = LevelScript(world, aid, v, dry)
    fn(ab)
    return ab.finalize((area_name(aid, v), music))


# ----------------------------------------------------------------------------
# /* src/game/save_file.c */  SaveFile (session memory only - FILES = OFF)
# ----------------------------------------------------------------------------
class SaveFile:  # /* src/game/save_file.h */
    def __init__(self, seed):
        self.seed = seed
        self.discovered = set()
        self.visits = Counter()
        self.stars = set()
        self.flags = set()
        self.portal_uses = Counter()
        self.edges = set()
        self.actions = Counter()
        self.secrets = set()
        self.known_exits = {}
        self.node_names = {}
        self.coins = 0
        self.lives = 4
        self.total = 0
        self.time = 0.0
        self.ended = False
        rng = random.Random(seed_hash(seed, 'rare'))
        cands = list(RARE_HOST_CANDIDATES)
        rng.shuffle(cands)
        self.rare_hosts = {}
        for i, rid in enumerate(RARE_AREAS):
            self.rare_hosts[cands[i]] = (rid, rng.randint(2, 4))

    def has(self, f):
        return f in self.flags

    def n(self):
        return len(self.stars)

    def count_all(self):
        """Dry-build every area variant to count Power Stars / stars (yields progress)."""
        items = [(aid, v) for aid, d in AREAS.items() for v in range(d[1])]
        tot = 0
        for i, (aid, v) in enumerate(items):
            a = build_area(self, aid, v, dry=True)
            tot += len(a.stars)
            if i % 4 == 0:
                yield i / len(items)
        self.total = tot
        yield 1.0


# ----------------------------------------------------------------------------
# /* actors/group / dialogue */  B3313 NPC lines (built-in strings only)
# ----------------------------------------------------------------------------
STRANGE = [
    "I OPENED A DOOR YESTERDAY AND IT OPENED ME BACK.", "THE CASTLE HAS MORE ROOMS AT NIGHT. NOBODY KNOWS WHERE THEY GO.",
    "SOMETIMES I HEAR A SECOND SET OF FOOTSTEPS WHEN I WALK ALONE.", "HAVE YOU NOTICED THE PAINTINGS BLINK?",
    "THIS HALLWAY USED TO BE SHORTER. OR I USED TO BE LONGER.", "IF YOU FALL ASLEEP IN THE CASTLE YOU WAKE UP IN A DIFFERENT ONE.",
    "I'M ALMOST SURE I'VE HAD THIS CONVERSATION WITH YOU BEFORE.", "THE WALLS HERE ARE PAPER-THIN. SOME ARE JUST PAPER.",
    "THE SKY WAS A DIFFERENT COLOUR WHEN I ARRIVED. I DON'T REMEMBER WHICH.",
    "DON'T TRUST DOORS THAT OPEN THE SAME WAY TWICE.", "THERE'S A ROOM WITH NOTHING IN IT. IT'S THE LOUDEST ROOM HERE.",
]
HINTS = [
    "HAVE YOU SEEN THE {a}? IT'S PAST A DOOR YOU ALREADY OPENED.", "SOMEONE SAID THE {a} IS HIDDEN BEHIND A PAINTING THAT MOVED.",
    "THE {a}... I WENT THERE ONCE. I THINK. IT WAS UPSIDE DOWN.", "IF YOU FIND THE {a}, TELL IT I SAID HELLO.",
    "THE ROAD TO THE {a} STARTS SOMEWHERE WITH WATER. OR STAIRS. ONE OF THOSE.",
]
MISLEAD = [
    "ALL THE POWER STARS ARE UNDER WATER. ALL OF THEM. TRUST ME.", "THE FRONT DOOR IS THE ONLY DOOR THAT GOES ANYWHERE REAL.",
    "IF YOU JUMP INTO A PAINTING BACKWARDS YOU COME OUT YESTERDAY.", "THERE ARE EXACTLY TWELVE ROOMS. I COUNTED TWICE.",
]
PORTAL_WORDS = {'left': "UPPER LEFT", 'right': "UPPER RIGHT", 'd6': "SIXTH", 'd3': "THIRD", 'door_c': "NORTH-WEST",
                'cellar': "CELLAR", 'down': "DOWNWARD", 'top': "TOP", 'lobby': "LOBBY", 'entry': "ENTRANCE",
                'cave': "CAVE", 'pipe': "GREEN PIPE"}


def npc_extra_lines(g, npc):
    w = g.world
    rng = random.Random(seed_hash(w.seed, npc.name, w.visits[g.area.aid], len(w.stars), len(w.discovered)))
    out = []
    known = {k.split('#')[0] for k in w.discovered}
    undisc = [AREAS[a][0] for a in AREAS if a not in RARE_AREAS and a not in known]
    n = len(w.stars)
    if n >= 30:
        out.append("YOU CARRY %d POWER STARS. THE LAST COURSE IS AWAKE NOW. THE TELESCOPE, THE RAINBOW, THE GLITCH..." % n)
    elif n >= 10:
        out.append("YOU'RE GLOWING. %d POWER STARS. THE CASTLE HAS STARTED NOTICING YOU." % n)
    elif n == 0:
        out.append("NO POWER STARS YET? THEY SPARKLE. THEY SPIN. YOU CAN'T MISS THEM. PEOPLE MISS THEM.")
    if undisc and rng.random() < 0.85:
        out.append(rng.choice(HINTS).format(a=rng.choice(undisc)))
    if rng.random() < 0.3:
        hosts = list(w.rare_hosts.items())
        (ha, hv, hp), (rid, k) = hosts[rng.randrange(len(hosts))]
        out.append("A RUMOUR: IN THE %s, THE %s DOOR ISN'T ITSELF THE %s TIME YOU USE IT." %
                   (area_name(ha, hv), PORTAL_WORDS.get(hp, hp.upper()), ("", "FIRST", "SECOND", "THIRD", "FOURTH")[k]))
    if rng.random() < 0.5:
        out.append(rng.choice(STRANGE))
    if rng.random() < 0.22:
        out.append(rng.choice(MISLEAD))
    return out


# ----------------------------------------------------------------------------
# /* actors/ */  entities (Power Stars, Mario NPCs, enemies)
# ----------------------------------------------------------------------------
class ObjectPickup:
    """Runtime pickup: Power Star (star) or Dream Mote (coin)."""
    __slots__ = ("kind", "x", "y", "z", "alive", "sid", "final", "cond", "label")

    def __init__(self, kind, x, y, z, sid=None, final=False, cond=None, label=None):
        self.kind, self.x, self.y, self.z = kind, x, y, z
        self.alive = True
        self.sid, self.final, self.cond, self.label = sid, final, cond, label


class NpcObject:
    # B3313 cast drawn as Mario NPCs: 0 toad 1 yoshi 2 luigi 3 boo 4 koopa 5 penguin 6 peach/metal
    STYLE_COL = {0: (240, 60, 60), 1: (80, 200, 70), 2: (40, 180, 70), 3: (240, 240, 250), 4: (60, 180, 60),
                 5: (240, 245, 255), 6: (255, 160, 200)}

    def __init__(self, d):
        (p, name, lines, style, yaw, col) = d
        self.x, self.y, self.z = p
        self.name = name
        self.lines = lines
        self.style = style
        self.yaw = yaw
        self.col = col or self.STYLE_COL.get(style, (200, 120, 90))
        self.t = random.random() * 10

    def update(self, dt, g):
        self.t += dt
        pl = g.player.pos
        dx, dz = pl.x - self.x, pl.z - self.z
        if dx * dx + dz * dz < 64:
            target = math.atan2(dx, dz)
            d = (target - self.yaw + math.pi) % TAU - math.pi
            self.yaw += d * min(1.0, dt * 4)


class ObjectEnemy:
    def __init__(self, d, rng):
        kind, p, rad, kw = d
        self.kind = kind
        self.home = p
        self.pos = Vec3f(*p)
        self.vy = 0.0
        self.rad = max(1.5, rad)
        self.hp = 2 if kind == 'blob' else 1
        self.alive = True
        self.dir = rng.random() * TAU
        self.t = rng.random() * 10
        self.axis = kw.get('axis', 0)
        self.sgn = 1
        default = {'blob': (150, 90, 50), 'spike': (230, 60, 90), 'cube': (120, 120, 140)}[kind]
        self.col = kw.get('col', default)
        self.r = {'blob': 0.75, 'spike': 0.7, 'cube': 1.0}[kind]
        self.h = {'blob': 1.1, 'spike': 1.2, 'cube': 1.8}[kind]
        self.squash = 0.0
        self.hover_y = p[1]
        self.chasing = False

    def update(self, dt, g):
        if not self.alive:
            return
        self.t += dt
        self.squash = max(0.0, self.squash - dt * 3)
        area = g.area
        pl = g.player.pos
        p = self.pos
        dx, dz = pl.x - p.x, pl.z - p.z
        d = math.hypot(dx, dz)
        dy = pl.y - p.y
        hx, hz = self.home[0] - p.x, self.home[2] - p.z
        if self.kind == 'cube':
            sp = 4.2
            if self.axis == 0:
                nx, nz = p.x + self.sgn * sp * dt, p.z
            else:
                nx, nz = p.x, p.z + self.sgn * sp * dt
            ahead_x = nx + (self.sgn * (self.r + 0.1) if self.axis == 0 else 0)
            ahead_z = nz + (self.sgn * (self.r + 0.1) if self.axis == 1 else 0)
            gy, _ = area.ground(ahead_x, ahead_z, p.y + 0.6)
            if area.point_solid(ahead_x, p.y + 0.9, ahead_z) or gy < p.y - 1.0 or \
                    math.hypot(nx - self.home[0], nz - self.home[2]) > self.rad:
                self.sgn = -self.sgn
            else:
                p.x, p.z = nx, nz
            self._fall(dt, area)
            return
        if self.kind == 'spike':
            self.chasing = d < 10 and abs(dy) < 5
            if self.chasing:
                self.dir = math.atan2(dx, dz)
                sp = 3.2
            else:
                self.dir += dt * 0.8
                sp = 1.6
                if math.hypot(hx, hz) > self.rad:
                    self.dir = math.atan2(hx, hz)
            nx = p.x + math.sin(self.dir) * sp * dt
            nz = p.z + math.cos(self.dir) * sp * dt
            ty = (pl.y + 0.8) if self.chasing else self.hover_y
            ty = clamp(ty, self.hover_y - 2.5, self.hover_y + 2.5)
            ny = p.y + (ty - p.y) * min(1.0, dt * 1.5) + math.sin(self.t * 3) * 0.01
            if not area.point_solid(nx, ny + 0.6, nz):
                p.x, p.z = nx, nz
            p.y = ny
            return
        # blob
        self.chasing = d < 9 and abs(dy) < 3
        if self.chasing:
            target = math.atan2(dx, dz)
            sp = 3.8
        else:
            target = self.dir + math.sin(self.t * 0.7) * 0.8
            sp = 1.3
            if math.hypot(hx, hz) > self.rad:
                target = math.atan2(hx, hz)
        dd = (target - self.dir + math.pi) % TAU - math.pi
        self.dir += dd * min(1.0, dt * 5)
        nx = p.x + math.sin(self.dir) * sp * dt
        nz = p.z + math.cos(self.dir) * sp * dt
        gy, _ = area.ground(nx, nz, p.y + 0.6)
        if gy < p.y - 1.4 or area.point_solid(nx, p.y + 0.6, nz) or \
                (area.water is not None and gy < area.water - 0.8):
            self.dir += math.pi * 0.6
        else:
            p.x, p.z = nx, nz
        self._fall(dt, area)

    def _fall(self, dt, area):
        p = self.pos
        gy, _ = area.ground(p.x, p.z, p.y + 0.6)
        self.vy -= GRAVITY * dt
        p.y += self.vy * dt
        if p.y <= gy:
            p.y = gy
            self.vy = 0.0
        if p.y < area.kill_y:
            self.alive = False


class MarioState:  # /* src/game/mario.h */
    def __init__(self):
        self.pos = Vec3f()
        self.vel = Vec3f()
        self.yaw = 0.0
        self.grounded = False
        self.ground = None
        self.hp = 8
        self.inv = 0.0
        self.state = 'normal'
        self.chain = 0
        self.chain_t = 0.0
        self.coyote = 0.0
        self.jbuf = 0.0
        self.pound_t = 0.0
        self.anim = 0.0
        self.crouch = False
        self.swim = False
        self.last_safe = (0.0, 0.0, 0.0)
        self.safe_t = 0.0
        self.squash = 0.0
        self.air_cap = RUN_SPEED
        self.flip = 0.0

    def place(self, x, y, z, yaw):
        self.pos.set(x, y, z)
        self.vel.set(0, 0, 0)
        self.yaw = yaw
        self.state = 'normal'
        self.grounded = False
        self.ground = None
        self.last_safe = (x, y, z)

    # -- movement -----------------------------------------------------------------
    def step(self, g, dt, inp):
        area = g.area
        au = g.audio
        p, v = self.pos, self.vel
        self.coyote -= dt
        self.chain_t -= dt
        self.inv = max(0.0, self.inv - dt)
        self.squash = max(0.0, self.squash - dt * 4)
        if inp.jump:
            self.jbuf = 0.13
        else:
            self.jbuf = max(0.0, self.jbuf - dt)
        hs = math.hypot(v.x, v.z)
        water = area.water is not None and p.y + 1.0 < area.water
        wx, wz, mag = inp.wx, inp.wz, inp.mag
        self.crouch = inp.crouch and self.grounded and not water
        if self.state == 'pound':
            self.pound_t -= dt
            v.x = v.z = 0.0
            if self.pound_t > 0:
                v.y = 0.0
            else:
                v.y = -30.0
        elif water:
            if not self.swim:
                au.play('splash', 0.6)
                v.y *= 0.3
            self.swim = True
            self.state = 'normal'
            tx, tz = wx * 4.6 * mag, wz * 4.6 * mag
            k = min(1.0, dt * 5)
            v.x += (tx - v.x) * k
            v.z += (tz - v.z) * k
            v.y -= 7.0 * dt
            if v.y < -3.5:
                v.y = -3.5
            if self.jbuf > 0:
                self.jbuf = 0
                near_top = p.y + 1.6 > area.water - 0.8
                v.y = 13.5 if near_top else 5.2
                au.play('jump', 0.5)
            if mag > 0.1:
                self._face(wx, wz, dt, 8)
        else:
            self.swim = False
            if self.grounded:
                maxs = CROUCH_SPEED if self.crouch else (RUN_SPEED if inp.run else WALK_SPEED)
                tx, tz = wx * maxs * mag, wz * maxs * mag
                acc = GROUND_ACC if mag > 0.1 else GROUND_DEC
                if self.crouch and hs > maxs:
                    acc = 16.0
                dx, dz = tx - v.x, tz - v.z
                dl = math.hypot(dx, dz)
                m = acc * dt
                if dl > m:
                    dx *= m / dl
                    dz *= m / dl
                v.x += dx
                v.z += dz
                if mag > 0.1:
                    self._face(wx, wz, dt, 16)
            else:
                if self.state != 'backflip' or v.y < 0:
                    v.x += wx * AIR_ACC * mag * dt
                    v.z += wz * AIR_ACC * mag * dt
                nh = math.hypot(v.x, v.z)
                if nh > self.air_cap and nh > 1e-6:
                    v.x *= self.air_cap / nh
                    v.z *= self.air_cap / nh
                if mag > 0.1 and self.state not in ('long', 'backflip'):
                    self._face(wx, wz, dt, 5)
            if self.jbuf > 0 and (self.grounded or self.coyote > 0):
                self.jbuf = 0.0
                fx, fz = math.sin(self.yaw), math.cos(self.yaw)
                if inp.crouch and hs < 3.0:
                    v.y = 17.4
                    v.x, v.z = -fx * 3.6, -fz * 3.6
                    self.state = 'backflip'
                    self.flip = 0.0
                    self.air_cap = 4.0
                    au.play('flip')
                    g.world.actions['backflips'] += 1
                elif inp.crouch and hs > 6.0:
                    sp = min(16.5, max(hs * 1.35, 13.5))
                    v.x, v.z = fx * sp, fz * sp
                    v.y = 9.4
                    self.state = 'long'
                    self.air_cap = sp
                    au.play('long')
                    g.world.actions['longjumps'] += 1
                else:
                    if self.chain_t > 0 and hs > 3.0:
                        self.chain = min(self.chain + 1, 2)
                    else:
                        self.chain = 0
                    v.y = JUMP_V[self.chain]
                    self.state = 'jump'
                    self.air_cap = max(hs, RUN_SPEED * 0.9)
                    au.play(('jump', 'jump2', 'jump3')[self.chain])
                    g.world.actions['jumps'] += 1
                self.grounded = False
                self.coyote = 0.0
                self.squash = 0.5
            elif not self.grounded and inp.crouch_edge and self.state not in ('pound', 'long'):
                self.state = 'pound'
                self.pound_t = 0.2
                v.x = v.z = 0.0
                v.y = 0.0
                au.play('flip', 0.4)
        if self.state == 'backflip':
            self.flip += dt * 9.0
        if not water and not (self.state == 'pound' and self.pound_t > 0):
            v.y -= GRAVITY * dt
            if v.y < TERMINAL:
                v.y = TERMINAL
        self._collide(g, area, dt)
        if self.grounded and self.ground is not None and self.ground.mover is None:
            self.safe_t += dt
            if self.safe_t > 0.4:
                self.safe_t = 0.0
                if area.lava is None or p.y > area.lava + 0.3:
                    self.last_safe = (p.x, p.y, p.z)
        self.anim += dt * (4 + hs * 1.3)

    def _face(self, wx, wz, dt, rate):
        target = math.atan2(wx, wz)
        d = (target - self.yaw + math.pi) % TAU - math.pi
        self.yaw += d * min(1.0, dt * rate)

    def _collide(self, g, area, dt):
        p, v = self.pos, self.vel
        r = P_RADIUS
        if self.grounded and self.ground is not None and self.ground.mover is not None:
            m = self.ground.mover
            p.x += m.dx
            p.y += m.dy
            p.z += m.dz
        p.x += v.x * dt
        p.z += v.z * dt
        cols = area.query(p.x, p.z, r + 1.2)
        for _ in range(2):
            for c in cols:
                k = c.k
                if k == 3:
                    continue
                if k == 4:
                    if p.y + P_HEIGHT < c.y0 or p.y > c.y1:
                        continue
                    dx, dz = p.x - c.cx, p.z - c.cz
                    d = math.hypot(dx, dz)
                    lim = c.r - r
                    if d > lim and d > 1e-6:
                        ux, uz = dx / d, dz / d
                        p.x, p.z = c.cx + ux * lim, c.cz + uz * lim
                        vn = v.x * ux + v.z * uz
                        if vn > 0:
                            v.x -= ux * vn
                            v.z -= uz * vn
                    continue
                if p.y + P_HEIGHT <= c.y0 + 0.02:
                    continue
                if k == 2:
                    if p.y + STEP_H >= c.y1:
                        continue
                    dx, dz = p.x - c.cx, p.z - c.cz
                    d2 = dx * dx + dz * dz
                    lim = c.r + r
                    if d2 < lim * lim:
                        d = math.sqrt(d2)
                        if d < 1e-6:
                            dx, dz, d = 1.0, 0.0, 1.0
                        ux, uz = dx / d, dz / d
                        p.x, p.z = c.cx + ux * lim, c.cz + uz * lim
                        vn = v.x * ux + v.z * uz
                        if vn < 0:
                            v.x -= ux * vn
                            v.z -= uz * vn
                    continue
                nx = c.x0 if p.x < c.x0 else c.x1 if p.x > c.x1 else p.x
                nz = c.z0 if p.z < c.z0 else c.z1 if p.z > c.z1 else p.z
                top = c.top(nx, nz) if k == 1 else c.y1
                if p.y + STEP_H >= top:
                    continue
                dx, dz = p.x - nx, p.z - nz
                d2 = dx * dx + dz * dz
                if d2 >= r * r:
                    continue
                if d2 > 1e-10:
                    d = math.sqrt(d2)
                    ux, uz = dx / d, dz / d
                    p.x, p.z = nx + ux * r, nz + uz * r
                    vn = v.x * ux + v.z * uz
                    if vn < 0:
                        v.x -= ux * vn
                        v.z -= uz * vn
                else:
                    pens = ((p.x - c.x0, 0), (c.x1 - p.x, 1), (p.z - c.z0, 2), (c.z1 - p.z, 3))
                    m = min(pens)[1]
                    if m == 0:
                        p.x = c.x0 - r
                        v.x = min(v.x, 0)
                    elif m == 1:
                        p.x = c.x1 + r
                        v.x = max(v.x, 0)
                    elif m == 2:
                        p.z = c.z0 - r
                        v.z = min(v.z, 0)
                    else:
                        p.z = c.z1 + r
                        v.z = max(v.z, 0)
        y_prev = p.y
        p.y += v.y * dt
        if v.y > 0:
            for c in cols:
                if c.k in (3, 4):
                    continue
                if c.contains(p.x, p.z, -0.08) and y_prev + P_HEIGHT - 0.05 <= c.y0 < p.y + P_HEIGHT:
                    p.y = c.y0 - P_HEIGHT
                    v.y = 0.0
                    if self.state == 'backflip':
                        self.state = 'jump'
        gy = -1e9
        gc = None
        lim = max(y_prev, p.y) + STEP_H
        for c in cols:
            if c.k == 4 or not c.contains(p.x, p.z, r * 0.55):
                continue
            t = c.top(p.x, p.z)
            if t <= lim and t > gy:
                gy = t
                gc = c
        was = self.grounded
        if v.y <= 0 and p.y <= gy:
            p.y = gy
            if not was:
                self._land(g, v.y)
            v.y = 0.0
            self.grounded = True
        elif was and v.y <= 0 and p.y - gy < 0.45:
            p.y = gy
            v.y = 0.0
            self.grounded = True
        else:
            self.grounded = False
        self.ground = gc if self.grounded else None
        if was and not self.grounded and v.y <= 0:
            self.coyote = 0.1
            if self.state == 'normal':
                self.air_cap = max(math.hypot(v.x, v.z), WALK_SPEED)

    def _land(self, g, vy):
        if self.state == 'pound':
            g.pound_impact()
        if self.state == 'jump':
            self.chain_t = 0.28
        else:
            self.chain = 0
        if self.state == 'long':
            self.vel.x *= 0.5
            self.vel.z *= 0.5
        self.state = 'normal'
        self.squash = min(1.0, -vy / 25.0)


class Controllers:  # /* src/game/game_init Controllers */
    __slots__ = ("wx", "wz", "mag", "run", "crouch", "jump", "crouch_edge")

    def __init__(self):
        self.wx = self.wz = self.mag = 0.0
        self.run = self.crouch = self.jump = self.crouch_edge = False


class GameSettings:  # /* save flags / options */
    def __init__(self):
        self.fps = 60
        self.fog = True
        self.audio = True
        self.music = True
        self.render_dist = 110
        self.sens = 1.0
        self.jitter = True
        self.capture = False
        self.debug = False
        self.seed = 1997


# ----------------------------------------------------------------------------
# /* src/game/main.c */  B3313Engine (cat's b3313)
# ----------------------------------------------------------------------------
class B3313Engine:
    FPS_CHOICES = (30, 60, 120)

    def __init__(self, headless=False):
        pygame.mixer.pre_init(22050, -16, 2, 512)
        pygame.init()
        pygame.display.set_caption(TITLE + "  -  " + VERSION + "  -  FILES = OFF")
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        self.clock = pygame.time.Clock()
        self.font = Font()
        self.settings = GameSettings()
        self.audio = Audio()
        self.r = GraphRenderer()
        self.cam = Camera()
        self.player = MarioState()
        self.world = None
        self.area = None
        self.state = 'loading'
        self.menu_i = 0
        self.prev_state = 'title'
        self.msgs = []
        self.banner = None
        self.banner_t = 0.0
        self.t = 0.0
        self.acc = 0.0
        self.running = True
        self.fade = 0.0
        self.trans = None
        self.dialog = None
        self.enemies = []
        self.npcs = []
        self.items = []
        self.switch_objs = []
        self.shake = 0.0
        self.jump_edge = False
        self.crouch_edge = False
        self.talk_edge = False
        self.mouse_dx = 0.0
        self.mouse_dy = 0.0
        self.cam_input_t = 0.0
        self.cur_node = None
        self.spawn_point = (0, 0, 0, 0)
        self.map_cache = None
        self.title_area = None
        self.title_world = None
        self.loader = self._load()
        self.load_p = 0.0
        self.load_msg = "LOADING AUDIO (B3313)"
        self.overlay = pygame.Surface((RW, RH), pygame.SRCALPHA)
        self.dim = pygame.Surface((RW, RH), pygame.SRCALPHA)
        self.dim.fill((0, 0, 10, 170))
        self.scaled_size = (RW * 3, RH * 3)
        self.blit_off = ((WIN_W - RW * 3) // 2, (WIN_H - RH * 3) // 2)
        self.ending_t = 0.0
        self.hires = pygame.Surface((WIN_W, WIN_H))

    # ------------------------------------------------------------------ loading
    def _load(self):
        for p in self.audio.build(self.settings.seed):
            self.load_p = p * 0.6
            yield
        self.load_msg = "LOADING B3313 LEVEL SCRIPTS"
        self.title_world = SaveFile(self.settings.seed)
        self.title_area = build_area(self.title_world, 'grounds', 0)
        self.load_p = 0.8
        yield
        self.load_msg = "COUNTING POWER STARS"
        for p in self.title_world.count_all():
            self.load_p = 0.8 + p * 0.2
            yield
        self.state = 'title'
        self.menu_i = 0
        self.audio.music('dream')

    # ------------------------------------------------------------------ session
    def new_game(self):
        w = SaveFile(self.settings.seed)
        for _ in w.count_all():
            pass
        self.world = w
        self.player = MarioState()
        self.player.hp = 8
        self.msgs = []
        self.enter_area('grounds', 0, 'entry', None)
        self.say("PROGRESS IS KEPT IN MEMORY FOR THIS SESSION ONLY (FILES = OFF).", 5)
        self.state = 'play'

    def say(self, text, dur=3.0):
        self.msgs.append([text, dur])
        if len(self.msgs) > 4:
            self.msgs.pop(0)

    def enter_area(self, aid, v, tag, from_node):
        w = self.world
        a = build_area(w, aid, v)
        self.area = a
        sp = a.spawns.get(tag) or a.spawns.get('entry')
        if sp is None:
            if a.portals:
                ex = a.portals[0].exit_point()
                sp = (ex[0], ex[1], ex[2], a.portals[0].yaw)
            else:
                sp = (0.0, 1.0, 0.0, 0.0)
        self.spawn_point = sp
        self.player.place(sp[0], sp[1] + 0.05, sp[2], sp[3])
        self.cam.snap(sp[0], sp[1], sp[2], sp[3])
        self._populate(a)
        w.visits[aid] += 1
        node = node_key(aid, a.v)
        w.node_names[node] = a.name
        new = node not in w.discovered
        w.discovered.add(node)
        if from_node and from_node != node:
            w.edges.add((from_node, node))
        self.cur_node = node
        self._record_exits()
        self.banner = a.name
        self.banner_t = 3.5
        if new:
            if aid in RARE_AREAS:
                self.audio.play('secret')
                self.say("A RARE ROOM. FEW EXPLORERS EVER SEE THIS.", 4)
                w.secrets.add('rare_' + aid)
            else:
                self.say("NEW AREA DISCOVERED", 2.5)
        self.audio.music(a.music)
        self.map_cache = None

    def _populate(self, a, keep_items=None):
        rng = random.Random(a.seed)
        w = self.world
        self.enemies = [ObjectEnemy(d, rng) for d in a.enemies]
        self.npcs = [NpcObject(d) for d in a.npcs]
        items = []
        for s in a.stars:
            if s.sid in w.stars:
                continue
            items.append(ObjectPickup('star', s.x, s.y, s.z, s.sid, s.final, s.cond, s.label))
        for i, m in enumerate(a.coins):
            c = ObjectPickup('coin', m[0], m[1], m[2])
            if keep_items is not None and i in keep_items:
                c.alive = False
            items.append(c)
        self.items = items
        pl = self.player.pos
        for pt in a.portals:
            pt.armed = not pt.contains(pl.x, pl.y, pl.z)

    def _record_exits(self):
        w = self.world
        outs = set()
        for pt in self.area.portals:
            if pt.is_visible(w):
                d = pt.dest(w)
                if d:
                    outs.add(node_key(d[0], clamp(d[1], 0, AREAS[d[0]][1] - 1)))
        w.known_exits[self.cur_node] = outs

    def rebuild_area(self):
        a = self.area
        dead = {i for i, it in enumerate([x for x in self.items if x.kind == 'coin']) if not it.alive}
        pl = self.player
        pos = (pl.pos.x, pl.pos.y, pl.pos.z)
        self.area = build_area(self.world, a.aid, a.v)
        self._populate(self.area, keep_items=dead)
        pl.pos.set(*pos)
        pl.ground = None
        self._record_exits()
        self.map_cache = None

    # ------------------------------------------------------------------ events
    def use_portal(self, pt):
        w = self.world
        if pt.is_locked(w):
            self.say(pt.lock[1], 3)
            self.audio.play('lock')
            pl = self.player
            pl.vel.x, pl.vel.z = pt.fx * 7, pt.fz * 7
            pl.vel.y = 5
            pt.armed = False
            return
        d = pt.dest(w)
        if d is None:
            return
        w.portal_uses[pt.pid] += 1
        w.actions['portals'] += 1
        if pt.pid == 'clock:0:top':
            w.flags.add('clock_done')
        self.audio.play('portal')
        self.trans = (d, self.cur_node, (d[0] == self.area.aid and d[1] == self.area.v))
        self.state = 'fade_out'
        self.fade = 0.0

    def pound_impact(self):
        self.shake = 0.25
        self.audio.play('pound')
        pl = self.player.pos
        for e in self.enemies:
            if e.alive and e.kind != 'cube':
                if math.hypot(e.pos.x - pl.x, e.pos.z - pl.z) < 3.2 and abs(e.pos.y - pl.y) < 2:
                    self.kill_enemy(e)

    def kill_enemy(self, e):
        e.alive = False
        self.audio.play('enemy')
        self.world.actions['defeated'] += 1
        for k in range(2):
            self.items.append(ObjectPickup('coin', e.pos.x + (k - 0.5), e.pos.y + 0.8, e.pos.z))

    def hurt(self, n, fx=0.0, fz=0.0):
        pl = self.player
        if pl.inv > 0:
            return
        pl.hp -= n
        pl.inv = 1.6
        pl.vel.x, pl.vel.z = fx * 8, fz * 8
        pl.vel.y = 7.5
        pl.state = 'normal'
        pl.grounded = False
        self.shake = 0.2
        self.audio.play('damage')
        self.world.actions['hits'] += 1
        if pl.hp <= 0:
            self.lose_life()

    def lose_life(self):
        w = self.world
        w.lives -= 1
        w.actions['deaths'] += 1
        pl = self.player
        if w.lives < 0:
            self.state = 'gameover'
            self.menu_i = 0
            return
        pl.hp = 8
        pl.inv = 2.0
        sp = self.spawn_point
        pl.place(sp[0], sp[1] + 0.05, sp[2], sp[3])
        self.cam.snap(sp[0], sp[1], sp[2], sp[3])
        self.say("MARIO TOOK DAMAGE... %d LIVES LEFT" % w.lives, 3)

    def collect(self, it):
        w = self.world
        it.alive = False
        if it.kind == 'coin':
            w.coins += 1
            self.player.hp = min(8, self.player.hp + 1)
            self.audio.play('coin', 0.7)
            if w.coins % 50 == 0:
                w.lives += 1
                self.audio.play('life')
                self.say("50 COINS: EXTRA LIFE!", 3)
            return
        w.stars.add(it.sid)
        self.player.hp = 8
        self.audio.play('star')
        self.say("POWER STAR!  %d / %d" % (len(w.stars), w.total), 3.5)
        if it.final:
            w.ended = True
            self.state = 'ending'
            self.ending_t = 0.0
            self.audio.play('ending')
            self.audio.music('dream')
        elif len(w.stars) in (4, 10, 15, 20, 30):
            self.say("SOMEWHERE, A LOCKED DOOR SHIFTS ITS WEIGHT.", 4)

    # ------------------------------------------------------------------ simulation
    def sim(self, dt, inp):
        w = self.world
        a = self.area
        pl = self.player
        self.t += dt
        w.time += dt
        a.update_movers(self.t)
        pl.step(self, dt, inp)
        inp.jump = False
        inp.crouch_edge = False
        p = pl.pos
        # hazards
        if a.lava is not None and p.y < a.lava + 0.08:
            p.y = a.lava + 0.08
            if pl.inv <= 0:
                self.hurt(2)
            pl.vel.y = 16.0
            pl.grounded = False
        if p.y < a.kill_y:
            self.hurt(1)
            if self.state == 'play':
                ls = pl.last_safe
                pl.place(ls[0], ls[1] + 0.1, ls[2], pl.yaw)
                self.cam.snap(ls[0], ls[1], ls[2], self.cam.yaw)
                self.say("FALL RECOVERY", 1.5)
        # portals
        for pt in a.portals:
            if not pt.is_visible(w):
                continue
            inside = pt.contains(p.x, p.y, p.z)
            if not pt.armed:
                if not inside:
                    pt.armed = True
                continue
            if inside:
                self.use_portal(pt)
                if self.state != 'play':
                    return
        # pickups
        mid = p.y + 0.8
        for it in self.items:
            if not it.alive:
                continue
            if it.cond is not None and not it.cond(w):
                continue
            dx, dy, dz = it.x - p.x, it.y - mid, it.z - p.z
            rr = 1.3 if it.kind == 'star' else 1.0
            if dx * dx + dz * dz < rr * rr and abs(dy) < 1.4:
                self.collect(it)
                if self.state != 'play':
                    return
        # switches
        for (sp, flag, msg) in a.switches:
            if flag in w.flags:
                continue
            if abs(p.x - sp[0]) < 0.95 and abs(p.z - sp[2]) < 0.95 and -0.3 < p.y - sp[1] < 0.8:
                w.flags.add(flag)
                w.actions['switches'] += 1
                self.audio.play('switch')
                self.say(msg, 4)
                self.rebuild_area()
                return
        # triggers
        for tr in a.triggers:
            if tr[0] <= p.x <= tr[3] and tr[1] - 0.5 <= p.y + 0.5 <= tr[4] + 0.5 and tr[2] <= p.z <= tr[5]:
                kind, data, msg = tr[6], tr[7], tr[8]
                if kind == 'secret' and data not in w.secrets:
                    w.secrets.add(data)
                    self.audio.play('secret')
                    self.say(msg or "SECRET FOUND", 3)
                elif kind == 'flag' and data not in w.flags:
                    w.flags.add(data)
                    if msg:
                        self.say(msg, 3)
        # enemies
        for e in self.enemies:
            if not e.alive:
                continue
            e.update(dt, self)
            ep = e.pos
            dx, dz = p.x - ep.x, p.z - ep.z
            d = math.hypot(dx, dz)
            if d < e.r + P_RADIUS and p.y < ep.y + e.h and p.y + P_HEIGHT > ep.y:
                stomp = pl.vel.y < 0 and p.y > ep.y + e.h * 0.45
                if stomp and (e.kind == 'blob' or (e.kind == 'spike' and pl.state == 'pound')):
                    e.hp -= 2 if pl.state == 'pound' else 1
                    e.squash = 1.0
                    pl.vel.y = 11.0
                    pl.state = 'jump'
                    self.audio.play('enemy', 0.6)
                    if e.hp <= 0:
                        self.kill_enemy(e)
                elif stomp and e.kind == 'cube':
                    pl.vel.y = 12.0
                    pl.state = 'jump'
                else:
                    ux, uz = (dx / d, dz / d) if d > 1e-6 else (0.0, 1.0)
                    self.hurt(1 if e.kind != 'spike' else 2, ux, uz)
        for n in self.npcs:
            n.update(dt, self)

    # ------------------------------------------------------------------ input
    def gather_input(self):
        k = pygame.key.get_pressed()
        inp = Controllers()
        ix = (k[pygame.K_d] or k[pygame.K_RIGHT]) - (k[pygame.K_a] or k[pygame.K_LEFT])
        iz = (k[pygame.K_w] or k[pygame.K_UP]) - (k[pygame.K_s] or k[pygame.K_DOWN])
        mag = min(1.0, math.hypot(ix, iz))
        if mag > 0:
            cy, sy = math.cos(self.cam.yaw), math.sin(self.cam.yaw)
            wx = ix * cy + iz * sy
            wz = -ix * sy + iz * cy
            l = math.hypot(wx, wz)
            inp.wx, inp.wz = wx / l, wz / l
        inp.mag = mag
        inp.run = bool(k[pygame.K_LSHIFT] or k[pygame.K_RSHIFT])
        inp.crouch = bool(k[pygame.K_z] or k[pygame.K_LCTRL] or k[pygame.K_c])
        inp.jump = self.jump_edge
        inp.crouch_edge = self.crouch_edge
        self.jump_edge = False
        self.crouch_edge = False
        return inp

    def camera_controls(self, dt):
        k = pygame.key.get_pressed()
        s = self.settings.sens
        cam = self.cam
        if k[pygame.K_q]:
            cam.yaw -= 2.4 * s * dt
        if k[pygame.K_e]:
            cam.yaw += 2.4 * s * dt
        if k[pygame.K_PAGEUP] or k[pygame.K_i]:
            cam.pitch += 1.3 * s * dt
        if k[pygame.K_PAGEDOWN] or k[pygame.K_k]:
            cam.pitch -= 1.3 * s * dt
        if self.mouse_dx or self.mouse_dy:
            cam.yaw += self.mouse_dx * 0.005 * s
            cam.pitch += self.mouse_dy * 0.004 * s
            self.mouse_dx = self.mouse_dy = 0.0

    def set_capture(self, on):
        self.settings.capture = on
        try:
            pygame.event.set_grab(on)
            pygame.mouse.set_visible(not on)
            if on:
                pygame.mouse.get_rel()
        except Exception:
            pass

    def handle_event(self, ev):
        st = self.state
        if ev.type == pygame.QUIT:
            self.running = False
            return
        if ev.type == pygame.MOUSEMOTION:
            if st == 'play' and (self.settings.capture or ev.buttons[2]):
                self.mouse_dx += ev.rel[0]
                self.mouse_dy += ev.rel[1]
            return
        if ev.type == pygame.MOUSEWHEEL and st == 'play':
            self.cam.dist = clamp(self.cam.dist - ev.y * 0.6, 3.5, 14.0)
            return
        if ev.type != pygame.KEYDOWN:
            return
        key = ev.key
        if st == 'loading':
            return
        if st == 'play':
            if self.dialog:
                if key in (pygame.K_f, pygame.K_RETURN, pygame.K_SPACE):
                    self.dialog[1] += 1
                    self.audio.play('talk')
                    if self.dialog[1] >= len(self.dialog[2]):
                        self.dialog = None
                elif key == pygame.K_ESCAPE:
                    self.dialog = None
                return
            if key == pygame.K_ESCAPE:
                self.state = 'pause'
                self.menu_i = 0
                self.audio.play('menu')
                if self.settings.capture:
                    pygame.event.set_grab(False)
                    pygame.mouse.set_visible(True)
            elif key == pygame.K_SPACE:
                self.jump_edge = True
            elif key in (pygame.K_z, pygame.K_LCTRL, pygame.K_c):
                self.crouch_edge = True
            elif key == pygame.K_r:
                sp = self.spawn_point
                self.player.place(sp[0], sp[1] + 0.05, sp[2], sp[3])
                self.cam.snap(sp[0], sp[1], sp[2], sp[3])
                self.say("RESPAWN", 1.2)
            elif key == pygame.K_f:
                n = self.nearest_npc()
                if n:
                    self.dialog = [n, 0, list(n.lines) + npc_extra_lines(self, n)]
                    self.audio.play('talk')
            elif key == pygame.K_TAB:
                self.prev_state = 'play'
                self.state = 'map'
            elif key == pygame.K_m:
                self.set_capture(not self.settings.capture)
            elif key == pygame.K_F3:
                self.settings.debug = not self.settings.debug
            elif key == pygame.K_F2:
                self.settings.jitter = not self.settings.jitter
            return
        if st in ('map', 'controls', 'about'):
            if key in (pygame.K_ESCAPE, pygame.K_RETURN, pygame.K_TAB, pygame.K_SPACE):
                self.state = self.prev_state
                self.audio.play('menu')
                if self.state == 'play' and self.settings.capture:
                    self.set_capture(True)
            return
        if st == 'gameover':
            if key in (pygame.K_RETURN, pygame.K_SPACE):
                self.world.lives = 4
                self.player.hp = 8
                self.enter_area('castle', 0, 'entry', self.cur_node)
                self.state = 'play'
            return
        if st == 'ending':
            if key in (pygame.K_RETURN, pygame.K_SPACE, pygame.K_ESCAPE) and self.ending_t > 2.0:
                self.enter_area('castle', 0, 'front', self.cur_node)
                self.state = 'play'
                self.say("THE CASTLE IS STILL HERE. IT ALWAYS IS.", 4)
            return
        items = self.menu_items()
        up = key in (pygame.K_UP, pygame.K_w)
        down = key in (pygame.K_DOWN, pygame.K_s)
        if up or down:
            self.menu_i = (self.menu_i + (1 if down else -1)) % len(items)
            self.audio.play('menu')
            return
        if key in (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_a, pygame.K_d) and st == 'settings':
            self.adjust_setting(items[self.menu_i], 1 if key in (pygame.K_RIGHT, pygame.K_d) else -1)
            return
        if key in (pygame.K_RETURN, pygame.K_SPACE):
            self.audio.play('select')
            self.menu_select(items[self.menu_i])
            return
        if key == pygame.K_ESCAPE:
            self.audio.play('menu')
            if st == 'pause':
                self.state = 'play'
                if self.settings.capture:
                    self.set_capture(True)
            elif st == 'settings':
                self.state = self.prev_state
                self.menu_i = 0
            elif st == 'title':
                pass

    def nearest_npc(self):
        p = self.player.pos
        best = None
        bd = 9.0
        for n in self.npcs:
            d = (n.x - p.x) ** 2 + (n.z - p.z) ** 2
            if d < bd and abs(n.y - p.y) < 2.5:
                bd = d
                best = n
        return best

    # ------------------------------------------------------------------ menus
    def menu_items(self):
        st = self.state
        if st == 'title':
            return ['START', 'CONTINUE', 'SETTINGS', 'CONTROLS', 'ABOUT', 'QUIT']
        if st == 'pause':
            return ['RESUME', 'AREA MAP', 'CONTROLS', 'SETTINGS', 'RETURN TO HUB', 'QUIT TO TITLE']
        if st == 'settings':
            return ['FPS CAP', 'FOG', 'AUDIO', 'MUSIC', 'RENDER DISTANCE', 'CAMERA SENSITIVITY', 'PS1 JITTER',
                    'MOUSE CAPTURE', 'DEBUG INFO', 'WORLD SEED', 'BACK']
        return ['BACK']

    def setting_value(self, name):
        s = self.settings
        on = lambda b: "ON" if b else "OFF"
        return {'FPS CAP': str(s.fps), 'FOG': on(s.fog), 'AUDIO': on(s.audio) + ("" if self.audio.ok else " (NO DEVICE)"),
                'MUSIC': on(s.music), 'RENDER DISTANCE': str(s.render_dist), 'CAMERA SENSITIVITY': "%.1f" % s.sens,
                'PS1 JITTER': on(s.jitter), 'MOUSE CAPTURE': on(s.capture), 'DEBUG INFO': on(s.debug),
                'WORLD SEED': str(s.seed)}.get(name, "")

    def adjust_setting(self, name, d):
        s = self.settings
        self.audio.play('menu')
        if name == 'FPS CAP':
            i = self.FPS_CHOICES.index(s.fps) if s.fps in self.FPS_CHOICES else 1
            s.fps = self.FPS_CHOICES[(i + d) % 3]
        elif name == 'FOG':
            s.fog = not s.fog
        elif name == 'AUDIO':
            s.audio = not s.audio
            self.audio.sfx_on = s.audio
            if not s.audio:
                self.audio.stop_music()
            elif s.music:
                self.audio.music(self.area.music if (self.area and self.world) else 'dream')
        elif name == 'MUSIC':
            s.music = not s.music
            self.audio.music_on = s.music and s.audio
            if self.audio.music_on:
                self.audio.music(self.area.music if (self.area and self.world) else 'dream')
            else:
                self.audio.stop_music()
        elif name == 'RENDER DISTANCE':
            s.render_dist = int(clamp(s.render_dist + d * 20, 40, 240))
        elif name == 'CAMERA SENSITIVITY':
            s.sens = round(clamp(s.sens + d * 0.1, 0.2, 3.0), 1)
        elif name == 'PS1 JITTER':
            s.jitter = not s.jitter
        elif name == 'MOUSE CAPTURE':
            s.capture = not s.capture
        elif name == 'DEBUG INFO':
            s.debug = not s.debug
        elif name == 'WORLD SEED':
            s.seed = max(0, s.seed + d)

    def menu_select(self, item):
        st = self.state
        if st == 'title':
            if item == 'START':
                self.new_game()
            elif item == 'CONTINUE':
                if self.world is not None:
                    self.state = 'play'
                    self.audio.music(self.area.music)
                else:
                    self.say("NO SESSION IN MEMORY YET. (FILES = OFF)", 2)
                    self.audio.play('lock')
            elif item == 'SETTINGS':
                self.prev_state = 'title'
                self.state = 'settings'
                self.menu_i = 0
            elif item == 'CONTROLS':
                self.prev_state = 'title'
                self.state = 'controls'
            elif item == 'ABOUT':
                self.prev_state = 'title'
                self.state = 'about'
            elif item == 'QUIT':
                self.running = False
        elif st == 'pause':
            if item == 'RESUME':
                self.state = 'play'
                if self.settings.capture:
                    self.set_capture(True)
            elif item == 'AREA MAP':
                self.prev_state = 'pause'
                self.state = 'map'
            elif item == 'CONTROLS':
                self.prev_state = 'pause'
                self.state = 'controls'
            elif item == 'SETTINGS':
                self.prev_state = 'pause'
                self.state = 'settings'
                self.menu_i = 0
            elif item == 'RETURN TO HUB':
                self.enter_area('castle', 0, 'entry', self.cur_node)
                self.state = 'play'
            elif item == 'QUIT TO TITLE':
                self.state = 'title'
                self.menu_i = 1
                self.audio.music('dream')
                self.set_capture(False)
        elif st == 'settings':
            if item == 'BACK':
                self.state = self.prev_state
                self.menu_i = 0
                if self.state == 'pause':
                    self.menu_i = 3
            else:
                self.adjust_setting(item, 1)

    # ------------------------------------------------------------------ update
    def update(self, dt):
        st = self.state
        if st == 'loading':
            try:
                next(self.loader)
            except StopIteration:
                self.state = 'title'
            return
        for m in self.msgs:
            m[1] -= dt
        self.msgs = [m for m in self.msgs if m[1] > 0]
        self.banner_t = max(0.0, self.banner_t - dt)
        self.shake = max(0.0, self.shake - dt)
        if st == 'play':
            if self.dialog:
                self.cam.update(self.player.pos.x, self.player.pos.y, self.player.pos.z, self.area, dt)
                for n in self.npcs:
                    n.update(dt, self)
                return
            self.camera_controls(dt)
            inp = self.gather_input()
            self.acc += dt
            steps = 0
            while self.acc >= SIM_DT and steps < MAX_STEPS and self.state == 'play':
                self.sim(SIM_DT, inp)
                self.acc -= SIM_DT
                steps += 1
            if steps >= MAX_STEPS:
                self.acc = 0.0
            if self.state == 'play' or self.state == 'fade_out':
                p = self.player.pos
                self.cam.update(p.x, p.y, p.z, self.area, dt)
        elif st == 'fade_out':
            self.fade += dt / 0.35
            if self.fade >= 1.0:
                (aid, v, tag), frm, same = self.trans
                self.enter_area(aid, v, tag, frm)
                if same:
                    self.world.actions['loops'] += 1
                    self.say("...AGAIN?" if aid != 'stairs' else "THE STAIRS CONTINUE...", 2.5)
                self.state = 'fade_in'
                self.fade = 1.0
        elif st == 'fade_in':
            self.fade -= dt / 0.35
            p = self.player.pos
            self.cam.update(p.x, p.y, p.z, self.area, dt)
            if self.fade <= 0:
                self.fade = 0.0
                self.state = 'play'
        elif st == 'ending':
            self.ending_t += dt

    # ------------------------------------------------------------------ drawing: world
    def draw_world(self, area, cam, t, entities=True):
        r = self.r
        s = self.settings
        r.begin(cam, area, s, t)
        wob = t * 2.0 if area.wobble else 0.0
        r.draw_static(area.meshes, wob)
        for m in area.movers:
            if r.near_view(m.x, m.y, m.z, 3):
                r.box(m.x, m.y, m.z, m.hx, m.hy, m.hz, m.color, bias=0.1)
        if entities and self.world is not None:
            self.draw_entities(area, t)
        r.flush()
        if area.water is not None and cam.pos.y < area.water:
            self.overlay.fill((*area.water_col, 110))
            r.surf.blit(self.overlay, (0, 0))
        if area.glitch and random.random() < 0.08:
            for _ in range(3):
                x, y = random.randrange(RW), random.randrange(RH)
                r.surf.fill(hsv(random.random(), 1, 1), (x, y, random.randint(8, 60), random.randint(1, 4)))

    def draw_entities(self, area, t):
        r = self.r
        w = self.world
        for pt in area.portals:
            if pt.is_visible(w) and r.near_view(pt.x, pt.y, pt.z, 4):
                self.draw_portal(pt, t)
        for it in self.items:
            if not it.alive or not r.near_view(it.x, it.y, it.z, 1.5):
                continue
            if it.cond is not None and not it.cond(w):
                continue
            if it.kind == 'star':
                bob = math.sin(t * 2.2 + it.x) * 0.15
                sc = 1.6 if it.final else 0.62
                # SM64 Power Star gold (final star brighter / warmer)
                col = (255, 245, 90) if it.final else (255, 220, 40)
                r.star(it.x, it.y + bob, it.z, sc, t * 2.4 + it.z, col)
            else:
                r.octa(it.x, it.y + math.sin(t * 3 + it.z) * 0.08, it.z, 0.2, 0.28, t * 3, (255, 220, 60), lit=True)
        for (sp, flag, msg) in area.switches:
            if not r.near_view(sp[0], sp[1], sp[2], 2):
                continue
            on = flag in w.flags
            r.box(sp[0], sp[1] + 0.1, sp[2], 0.8, 0.1, 0.8, (90, 90, 100))
            r.box(sp[0], sp[1] + (0.24 if not on else 0.16), sp[2], 0.5, 0.06 if on else 0.14, 0.5,
                  (80, 220, 90) if on else (230, 60, 60), bias=0.1)
        for n in self.npcs:
            if r.near_view(n.x, n.y + 1, n.z, 2):
                self.draw_npc(n, t)
        for e in self.enemies:
            if e.alive and r.near_view(e.pos.x, e.pos.y + 1, e.pos.z, 2):
                self.draw_enemy(e, t)
        self.draw_player(t)

    def draw_portal(self, pt, t):
        r = self.r
        w = self.world
        locked = pt.is_locked(w)
        d = pt.dest(w)
        h0 = (seed_hash(d[0] if d else 'x') % 1000) / 1000.0
        fx, fz = pt.fx, pt.fz
        rx, rz = fz, -fx
        k = pt.kind
        if k in ('door', 'painting'):
            nu, nv = (1, 3) if k == 'door' else (3, 2)
            off = 0.02 if k == 'door' else 0.1
            cx, cz = pt.x + fx * off, pt.z + fz * off
            W, H = pt.w, pt.h
            for j in range(nv):
                for i in range(nu):
                    l0, l1 = -W / 2 + W * i / nu, -W / 2 + W * (i + 1) / nu
                    y0, y1 = pt.y + H * j / nv, pt.y + H * (j + 1) / nv
                    if locked:
                        c = cmul((110, 110, 120), 0.8 + 0.2 * math.sin(t * 2 + j))
                    else:
                        c = hsv(h0 + 0.06 * math.sin(t * 1.4 + i * 1.3 + j * 1.9), 0.55 if k == 'door' else 0.7,
                                0.75 + 0.2 * math.sin(t * 2.1 + i + j * 2))
                    pts = [(cx + rx * l0, y0, cz + rz * l0), (cx + rx * l1, y0, cz + rz * l1),
                           (cx + rx * l1, y1, cz + rz * l1), (cx + rx * l0, y1, cz + rz * l0)]
                    r.face(pts, c, (fx, 0.0, fz), 0.35)
        elif k in ('hole', 'pipe'):
            y = pt.y + 0.03
            R = pt.w * (0.5 if k == 'hole' else 0.42)
            n = 8
            ring = [(pt.x + math.cos(TAU * i / n) * R, y, pt.z + math.sin(TAU * i / n) * R) for i in range(n)]
            r.face(ring[::-1], (10, 5, 20) if not locked else (60, 60, 60), (0, 1, 0), 0.4)
            if not locked:
                R2 = R * (0.55 + 0.15 * math.sin(t * 3))
                a0 = t * 1.5
                inner = [(pt.x + math.cos(a0 + TAU * i / 4) * R2, y + 0.02, pt.z + math.sin(a0 + TAU * i / 4) * R2)
                         for i in range(4)]
                r.face(inner[::-1], hsv(h0 + t * 0.1, 0.8, 1.0), (0, 1, 0), 0.5)
        else:
            R = pt.w * 0.5
            n = 10
            c0 = (pt.x, pt.y, pt.z)
            for i in range(n):
                a = TAU * i / n + t * 0.8
                b = TAU * (i + 1) / n + t * 0.8
                p1 = (pt.x + rx * math.cos(a) * R, pt.y + math.sin(a) * R, pt.z + rz * math.cos(a) * R)
                p2 = (pt.x + rx * math.cos(b) * R, pt.y + math.sin(b) * R, pt.z + rz * math.cos(b) * R)
                c = (90, 90, 100) if locked else hsv(h0 + i / n * 0.3 + t * 0.2, 0.6, 1.0)
                r.face([c0, p1, p2], c, None, 0.3)

    def _L(self, x, y, z, yaw):
        cy, sy = math.cos(yaw), math.sin(yaw)
        return lambda lx, ly, lz: (x + lx * cy + lz * sy, y + ly, z - lx * sy + lz * cy)

    def draw_player(self, t):
        """B3313 / Spaceworld '95 beta Mario — blocky red/blue N64 proportions."""
        pl = self.player
        r = self.r
        p = pl.pos
        area = self.area
        gy, _ = area.ground(p.x, p.z, p.y + 0.2)
        if gy > -1e8 and p.y - gy < 40:
            rad = clamp(0.5 - (p.y - gy) * 0.02, 0.2, 0.5)
            pts = [(p.x + math.cos(TAU * i / 8) * rad, gy + 0.04, p.z + math.sin(TAU * i / 8) * rad) for i in
                   range(8)]
            r.face(pts[::-1], (25, 22, 35), (0, 1, 0), 0.7)
        if pl.inv > 0 and int(pl.inv * 12) % 2 == 0:
            return
        yaw = pl.yaw
        hs = math.hypot(pl.vel.x, pl.vel.z)
        sq = 1.0 - pl.squash * 0.25
        if pl.crouch or pl.state == 'pound':
            sq *= 0.7
        L = self._L(p.x, p.y, p.z, yaw)
        swing = math.sin(pl.anim) * min(1.0, hs / 6.0) * 0.9 if pl.grounded else 0.5
        if pl.swim:
            swing = math.sin(t * 6) * 0.6
        body_p = 0.0
        if pl.state == 'long':
            body_p = 0.9
            swing = 0.0
        if pl.state == 'backflip':
            body_p = -pl.flip
        leg_p = -1.2 if pl.state == 'long' else 0.0
        # Spaceworld beta Mario palette
        red, blue, skin = (210, 28, 40), (28, 52, 180), (255, 198, 152)
        glove, shoe, hair, button = (248, 248, 248), (92, 48, 28), (72, 36, 20), (248, 208, 48)
        # legs (blue overalls) + brown shoes — stubbier beta proportions
        for sgn in (-1, 1):
            hip = L(sgn * 0.15, 0.52 * sq, 0)
            r.box(hip[0], hip[1], hip[2], 0.12, 0.22 * sq, 0.12, blue, yaw, swing * sgn + leg_p, (0, -0.22 * sq, 0))
            foot = L(sgn * 0.15, 0.12 * sq, 0.04)
            r.box(foot[0], foot[1], foot[2], 0.13, 0.08 * sq, 0.16, shoe, yaw, swing * sgn * 0.3 + leg_p * 0.2,
                  (0, -0.02, 0.06))
        # torso: blue overalls + red shirt
        c = L(0, 0.78 * sq, 0)
        r.box(c[0], c[1], c[2], 0.28, 0.26 * sq, 0.18, blue, yaw, body_p)
        c = L(0, 1.08 * sq, 0.01)
        r.box(c[0], c[1], c[2], 0.26, 0.14 * sq, 0.2, red, yaw, body_p)
        # yellow overall buttons (beta detail)
        for sgn in (-1, 1):
            b = L(sgn * 0.08, 0.92 * sq, 0.19)
            r.box(b[0], b[1], b[2], 0.04, 0.04, 0.02, button, yaw, body_p, bias=0.25, lit=False)
        # arms: red sleeves + white gloves
        for sgn in (-1, 1):
            sh = L(sgn * 0.38, 1.05 * sq, 0)
            ap = -swing * sgn if pl.state != 'pound' else -2.6
            if pl.state in ('jump', 'backflip') and not pl.grounded:
                ap = -2.4
            r.box(sh[0], sh[1], sh[2], 0.09, 0.18, 0.09, red, yaw, ap, (0, -0.18, 0))
            r.box(sh[0], sh[1], sh[2], 0.12, 0.1, 0.12, glove, yaw, ap, (0, -0.42, 0))
        # oversized beta head
        hc = L(0, 1.42 * sq, 0.02)
        r.box(hc[0], hc[1], hc[2], 0.28, 0.26, 0.26, skin, yaw, body_p * 0.5)
        # hair sides under cap
        for sgn in (-1, 1):
            h = L(sgn * 0.22, 1.48 * sq, -0.06)
            r.box(h[0], h[1], h[2], 0.06, 0.1, 0.1, hair, yaw, body_p * 0.5)
        # eyes + nose + mustache (Spaceworld face read)
        for sgn in (-1, 1):
            e = L(sgn * 0.1, 1.48 * sq, 0.28)
            r.box(e[0], e[1], e[2], 0.05, 0.07, 0.02, (250, 250, 250), yaw, 0, bias=0.25, lit=False)
            e = L(sgn * 0.1, 1.47 * sq, 0.3)
            r.box(e[0], e[1], e[2], 0.025, 0.04, 0.015, (20, 20, 40), yaw, 0, bias=0.3, lit=False)
        nose = L(0, 1.4 * sq, 0.3)
        r.box(nose[0], nose[1], nose[2], 0.07, 0.06, 0.06, skin, yaw, body_p * 0.4, bias=0.15)
        stache = L(0, 1.32 * sq, 0.3)
        r.box(stache[0], stache[1], stache[2], 0.14, 0.035, 0.04, hair, yaw, body_p * 0.4, bias=0.2, lit=False)
        # red cap + brim (no pom — beta Spaceworld hat)
        cc = L(0, 1.68 * sq, -0.02)
        r.box(cc[0], cc[1], cc[2], 0.3, 0.1, 0.28, red, yaw, body_p * 0.5)
        brim = L(0, 1.6 * sq, 0.18)
        r.box(brim[0], brim[1], brim[2], 0.22, 0.035, 0.12, red, yaw, body_p * 0.5)
        # white "M" patch on cap
        em = L(0, 1.72 * sq, 0.26)
        r.box(em[0], em[1], em[2], 0.08, 0.06, 0.02, (250, 250, 250), yaw, body_p * 0.5, bias=0.3, lit=False)

    def draw_npc(self, n, t):
        """B3313 NPCs drawn as Mario-cast (Toad, Yoshi, Luigi, Boo, Koopa, Penguin, Peach)."""
        r = self.r
        bob = math.sin(n.t * 2) * 0.04
        yaw = n.yaw
        L = self._L(n.x, n.y + bob, n.z, yaw)
        st = n.style
        skin = (255, 198, 152)
        head_y = 1.35

        if st == 0:  # Toad — mushroom head, vest
            red, vest, spots = (230, 40, 50), (40, 90, 220), (250, 250, 250)
            for sgn in (-1, 1):
                p = L(sgn * 0.12, 0.22, 0)
                r.box(p[0], p[1], p[2], 0.08, 0.2, 0.08, (250, 250, 250), yaw)
            p = L(0, 0.7, 0)
            r.box(p[0], p[1], p[2], 0.22, 0.28, 0.16, vest, yaw)
            p = L(0, 1.15, 0)
            r.box(p[0], p[1], p[2], 0.2, 0.18, 0.2, skin, yaw)
            p = L(0, 1.5, 0)
            r.box(p[0], p[1], p[2], 0.38, 0.28, 0.38, red, yaw)
            for sx, sz in ((-0.18, 0.2), (0.18, 0.2), (0.0, -0.22), (-0.22, -0.05), (0.22, -0.05)):
                q = L(sx, 1.55, sz)
                r.box(q[0], q[1], q[2], 0.08, 0.08, 0.08, spots, yaw, bias=0.2, lit=False)
            head_y = 1.15

        elif st == 1:  # Yoshi — green body, saddle, snout
            green, saddle, shoe = (80, 200, 70), (230, 40, 50), (230, 60, 60)
            for sgn in (-1, 1):
                p = L(sgn * 0.16, 0.2, 0.05)
                r.box(p[0], p[1], p[2], 0.1, 0.16, 0.14, shoe, yaw)
            p = L(0, 0.7, 0)
            r.box(p[0], p[1], p[2], 0.28, 0.35, 0.24, green, yaw)
            p = L(0, 1.05, 0.02)
            r.box(p[0], p[1], p[2], 0.3, 0.08, 0.26, saddle, yaw)
            p = L(0, 1.4, 0.05)
            r.box(p[0], p[1], p[2], 0.26, 0.24, 0.26, green, yaw)
            sn = L(0, 1.3, 0.32)
            r.box(sn[0], sn[1], sn[2], 0.14, 0.1, 0.16, green, yaw)
            for sgn in (-1, 1):
                e = L(sgn * 0.12, 1.55, 0.1)
                r.box(e[0], e[1], e[2], 0.05, 0.1, 0.05, (250, 250, 100), yaw, bias=0.2, lit=False)
            head_y = 1.4

        elif st == 2:  # Luigi — tall green/blue Mario brother
            green, blue, shoe = (40, 170, 60), (40, 70, 190), (50, 40, 40)
            for sgn in (-1, 1):
                p = L(sgn * 0.14, 0.35, 0)
                r.box(p[0], p[1], p[2], 0.1, 0.32, 0.1, blue, yaw)
                f = L(sgn * 0.14, 0.08, 0.06)
                r.box(f[0], f[1], f[2], 0.11, 0.07, 0.14, shoe, yaw)
            p = L(0, 0.95, 0)
            r.box(p[0], p[1], p[2], 0.24, 0.3, 0.16, blue, yaw)
            p = L(0, 1.28, 0)
            r.box(p[0], p[1], p[2], 0.22, 0.12, 0.18, green, yaw)
            for sgn in (-1, 1):
                sh = L(sgn * 0.32, 1.25, 0)
                r.box(sh[0], sh[1], sh[2], 0.08, 0.2, 0.08, green, yaw, 0, (0, -0.15, 0))
            p = L(0, 1.55, 0)
            r.box(p[0], p[1], p[2], 0.22, 0.22, 0.22, skin, yaw)
            stache = L(0, 1.42, 0.24)
            r.box(stache[0], stache[1], stache[2], 0.12, 0.03, 0.04, (40, 30, 20), yaw, bias=0.2, lit=False)
            cap = L(0, 1.78, -0.02)
            r.box(cap[0], cap[1], cap[2], 0.24, 0.08, 0.24, green, yaw)
            brim = L(0, 1.72, 0.16)
            r.box(brim[0], brim[1], brim[2], 0.18, 0.03, 0.1, green, yaw)
            head_y = 1.55

        elif st == 3:  # Boo — round white ghost, shy face
            white, tongue = (245, 245, 255), (230, 80, 100)
            p = L(0, 0.85, 0)
            r.box(p[0], p[1], p[2], 0.42, 0.42, 0.42, white, yaw)
            for sgn in (-1, 1):
                e = L(sgn * 0.14, 1.0, 0.4)
                r.box(e[0], e[1], e[2], 0.08, 0.1, 0.04, (20, 20, 40), yaw, bias=0.3, lit=False)
            m = L(0, 0.7, 0.42)
            r.box(m[0], m[1], m[2], 0.16, 0.08, 0.04, (20, 20, 40), yaw, bias=0.3, lit=False)
            tong = L(0, 0.62, 0.48)
            r.box(tong[0], tong[1], tong[2], 0.06, 0.08, 0.05, tongue, yaw, bias=0.25, lit=False)
            # tiny arms
            for sgn in (-1, 1):
                a = L(sgn * 0.4, 0.85, 0.05)
                r.box(a[0], a[1], a[2], 0.1, 0.08, 0.1, white, yaw)
            head_y = 0.85

        elif st == 4:  # Koopa Troopa — green shell, beak
            green, shell, beak, shoe = (70, 180, 70), (40, 140, 50), (240, 200, 60), (230, 60, 50)
            for sgn in (-1, 1):
                p = L(sgn * 0.14, 0.2, 0.05)
                r.box(p[0], p[1], p[2], 0.1, 0.16, 0.12, shoe, yaw)
            p = L(0, 0.65, 0)
            r.box(p[0], p[1], p[2], 0.22, 0.28, 0.18, green, yaw)
            p = L(0, 0.85, -0.12)
            r.box(p[0], p[1], p[2], 0.32, 0.28, 0.2, shell, yaw)
            p = L(0, 1.25, 0.05)
            r.box(p[0], p[1], p[2], 0.2, 0.18, 0.2, green, yaw)
            bk = L(0, 1.15, 0.28)
            r.box(bk[0], bk[1], bk[2], 0.1, 0.08, 0.12, beak, yaw)
            for sgn in (-1, 1):
                e = L(sgn * 0.1, 1.35, 0.18)
                r.box(e[0], e[1], e[2], 0.05, 0.06, 0.03, (250, 250, 100), yaw, bias=0.2, lit=False)
            head_y = 1.25

        elif st == 5:  # Penguin (SM64 Cool Cool Mountain)
            black, white, beak, feet = (30, 30, 45), (245, 245, 250), (250, 160, 40), (250, 140, 40)
            for sgn in (-1, 1):
                p = L(sgn * 0.14, 0.12, 0.08)
                r.box(p[0], p[1], p[2], 0.1, 0.08, 0.16, feet, yaw)
            p = L(0, 0.7, 0)
            r.box(p[0], p[1], p[2], 0.32, 0.4, 0.26, black, yaw)
            p = L(0, 0.65, 0.14)
            r.box(p[0], p[1], p[2], 0.22, 0.32, 0.08, white, yaw)
            p = L(0, 1.35, 0)
            r.box(p[0], p[1], p[2], 0.26, 0.24, 0.26, black, yaw)
            bk = L(0, 1.25, 0.3)
            r.box(bk[0], bk[1], bk[2], 0.08, 0.06, 0.12, beak, yaw)
            for sgn in (-1, 1):
                e = L(sgn * 0.1, 1.42, 0.24)
                r.box(e[0], e[1], e[2], 0.04, 0.05, 0.02, (250, 250, 250), yaw, bias=0.25, lit=False)
            head_y = 1.35

        else:  # 6 Peach (or metal/glitch tint for weird rooms)
            pink, dress, hair, crown = (255, 160, 200), (255, 120, 180), (250, 220, 100), (255, 215, 60)
            if 'METAL' in n.name or 'M3TAL' in n.name or 'SHADOW' in n.name:
                pink = dress = hair = (90, 100, 120)
                crown = (180, 190, 200)
                skin_c = (140, 150, 160)
            elif 'VANISH' in n.name:
                pink = dress = (200, 220, 255)
                skin_c = (220, 230, 255)
                hair = (180, 200, 255)
                crown = (200, 220, 255)
            else:
                skin_c = skin
            for sgn in (-1, 1):
                p = L(sgn * 0.12, 0.2, 0)
                r.box(p[0], p[1], p[2], 0.08, 0.18, 0.08, (250, 250, 250), yaw)
            p = L(0, 0.75, 0)
            r.box(p[0], p[1], p[2], 0.28, 0.4, 0.2, dress, yaw)
            p = L(0, 1.25, 0)
            r.box(p[0], p[1], p[2], 0.18, 0.12, 0.14, pink, yaw)
            p = L(0, 1.5, 0)
            r.box(p[0], p[1], p[2], 0.2, 0.2, 0.2, skin_c, yaw)
            p = L(0, 1.72, -0.02)
            r.box(p[0], p[1], p[2], 0.22, 0.1, 0.22, hair, yaw)
            cr = L(0, 1.88, 0)
            r.box(cr[0], cr[1], cr[2], 0.16, 0.06, 0.16, crown, yaw)
            head_y = 1.5

        # shared eyes for toad / others that didn't draw custom eyes above
        if st in (0, 2):
            for sgn in (-1, 1):
                e = L(sgn * 0.08, head_y + 0.02, 0.22)
                r.box(e[0], e[1], e[2], 0.035, 0.05, 0.02, (20, 20, 40), yaw, bias=0.25, lit=False)

        if self.nearest_npc() is n and not self.dialog:
            p = L(0, head_y + 0.85 + math.sin(t * 4) * 0.08, 0)
            r.star(p[0], p[1], p[2], 0.22, t * 2, (255, 230, 80))

    def draw_enemy(self, e, t):
        r = self.r
        p = e.pos
        yaw = e.dir
        if e.kind == 'blob':
            sq = 1.0 - e.squash * 0.5
            wob = math.sin(e.t * 9) * 0.08 if e.chasing else math.sin(e.t * 4) * 0.04
            L = self._L(p.x, p.y, p.z, yaw)
            c = L(0, 0.5 * sq, 0)
            r.box(c[0], c[1], c[2], 0.62, 0.45 * sq, 0.55, e.col, yaw, wob)
            c = L(0, 0.95 * sq, -0.05)
            r.box(c[0], c[1], c[2], 0.45, 0.14 * sq, 0.42, cmul(e.col, 1.15), yaw)
            for sgn in (-1, 1):
                q = L(sgn * 0.22, 0.65 * sq, 0.56)
                r.box(q[0], q[1], q[2], 0.1, 0.14, 0.02, (255, 255, 255), yaw, bias=0.2, lit=False)
                q = L(sgn * 0.2, 0.62 * sq, 0.58)
                r.box(q[0], q[1], q[2], 0.05, 0.08, 0.02, (10, 10, 10), yaw, bias=0.25, lit=False)
                q = L(sgn * 0.3, 0.1, math.sin(e.t * 10 + sgn) * 0.15)
                r.box(q[0], q[1], q[2], 0.16, 0.1, 0.22, (60, 40, 30), yaw)
        elif e.kind == 'spike':
            y = p.y + 0.6
            r.octa(p.x, y, p.z, 0.55, 0.7, e.t * 2, e.col)
            for i in range(4):
                a = e.t * 2 + i * HALF_PI + math.pi / 4
                r.octa(p.x + math.cos(a) * 0.6, y, p.z + math.sin(a) * 0.6, 0.12, 0.2, a, (240, 240, 250))
            L = self._L(p.x, y, p.z, yaw)
            q = L(0, 0.1, 0.42)
            r.box(q[0], q[1], q[2], 0.12, 0.12, 0.05, (255, 255, 120), yaw, bias=0.3, lit=False)
        else:
            L = self._L(p.x, p.y, p.z, yaw)
            c = L(0, 0.9, 0)
            r.box(c[0], c[1], c[2], 0.95, 0.9, 0.95, e.col, t * 0.6)
            for a in range(4):
                q = L(math.sin(a * HALF_PI + t * 0.6) * 0.97, 1.1, math.cos(a * HALF_PI + t * 0.6) * 0.97)
                r.box(q[0], q[1], q[2], 0.18, 0.18, 0.18, (255, 40, 40), 0, bias=0.2, lit=False)

    # ------------------------------------------------------------------ drawing: UI
    def _blit_px(self, surf, x, y, pixels, palette, scale=2):
        """Draw a tiny indexed pixel icon (Spaceworld HUD glyphs)."""
        for ry, row in enumerate(pixels):
            for rx, ch in enumerate(row):
                if ch == ' ' or ch not in palette:
                    continue
                surf.fill(palette[ch], (x + rx * scale, y + ry * scale, scale, scale))

    def _draw_sw_mario_head(self, surf, x, y, scale=2):
        # Spaceworld '95 lives icon — red cap, skin, mustache read
        px = [
            '  RRRRR  ',
            ' RRRRRRR ',
            'RRWWWRRRR',
            ' SSKKSS  ',
            'SSWOOWSS ',
            'SSNNNNSS ',
            ' SSSSS   ',
            '  S  S   ',
        ]
        pal = {'R': (210, 28, 40), 'W': (250, 250, 250), 'S': (255, 198, 152),
               'K': (72, 36, 20), 'O': (20, 20, 40), 'N': (72, 36, 20)}
        self._blit_px(surf, x, y, px, pal, scale)

    def _draw_sw_coin(self, surf, x, y, scale=2):
        px = [
            '  YYYY  ',
            ' YYYYYY ',
            'YY OO YY',
            'YY OO YY',
            'YY OO YY',
            ' YYYYYY ',
            '  YYYY  ',
        ]
        pal = {'Y': (248, 200, 40), 'O': (180, 120, 20)}
        self._blit_px(surf, x, y, px, pal, scale)

    def _draw_sw_star(self, surf, x, y, scale=2):
        px = [
            '    Y    ',
            '   YYY   ',
            'YYYYYYYYY',
            ' YYYYYYY ',
            '  YYYYY  ',
            ' YY   YY ',
            'Y       Y',
        ]
        pal = {'Y': (255, 230, 80)}
        self._blit_px(surf, x, y, px, pal, scale)

    def _draw_power_meter(self, surf, cx, cy, hp, R=14):
        """Spaceworld / SM64 power meter — 8 gold wedges around a hub."""
        pygame.draw.circle(surf, (12, 12, 20), (cx, cy), R + 3)
        pygame.draw.circle(surf, (40, 40, 55), (cx, cy), R + 1)
        for i in range(8):
            a0 = -HALF_PI + TAU * i / 8
            a1 = -HALF_PI + TAU * (i + 1) / 8
            pts = [(cx, cy)]
            for k in range(5):
                a = a0 + (a1 - a0) * k / 4
                pts.append((cx + math.cos(a) * R, cy + math.sin(a) * R))
            if hp > i:
                # classic warm yellow → orange as health drops
                if hp > 5:
                    col = (255, 220, 40)
                elif hp > 2:
                    col = (255, 160, 30)
                else:
                    col = (255, 60, 40)
            else:
                col = (28, 28, 38)
            pygame.draw.polygon(surf, col, pts)
            pygame.draw.polygon(surf, (8, 8, 16), pts, 1)
        pygame.draw.circle(surf, (18, 18, 28), (cx, cy), 4)
        pygame.draw.circle(surf, (255, 210, 60) if hp > 0 else (60, 60, 70), (cx, cy), 2)

    def draw_hud(self):
        """Spaceworld '95 SM64 HUD: Mario×lives, coin×N, star count, power meter."""
        s = self.r.surf
        f = self.font
        w = self.world
        pl = self.player
        # --- left column (Spaceworld stack) ---
        self._draw_sw_mario_head(s, 4, 3, 2)
        f.draw(s, "X%d" % max(0, w.lives), 26, 6, (255, 255, 255))
        self._draw_sw_coin(s, 6, 24, 2)
        f.draw(s, "X%d" % w.coins, 26, 28, (255, 230, 90))
        # power meter sits under lives/coins like the beta HUD wedge
        self._draw_power_meter(s, 18, 58, pl.hp, R=13)
        # --- top-right stars (Power Stars / Spaceworld star tally) ---
        self._draw_sw_star(s, RW - 58, 4, 2)
        f.draw(s, "%d" % len(w.stars), RW - 6, 6, (255, 240, 160), right=True)
        f.draw(s, "POWER STARS", RW - 6, 18, (200, 190, 150), right=True)
        f.draw(s, "/%d" % w.total, RW - 6, 28, (160, 150, 130), right=True)
        # course name + fps
        f.draw(s, self.area.name, 6, RH - 12, (230, 230, 255))
        f.draw(s, "%d FPS" % int(self.clock.get_fps() + 0.5), RW - 6, RH - 12, (180, 255, 180), right=True)
        if self.banner_t > 0 and self.banner:
            a = min(1.0, self.banner_t)
            col = lerpc((0, 0, 0), (255, 255, 255), a)
            f.draw(s, self.banner, RW // 2, 40, col, scale=2, center=True)
        y = RH - 34
        for m in reversed(self.msgs[-3:]):
            f.draw(s, m[0], RW // 2, y, (255, 255, 200), center=True)
            y -= 10
        n = self.nearest_npc()
        if n and not self.dialog:
            f.draw(s, "F: TALK TO " + n.name, RW // 2, RH - 48, (160, 255, 255), center=True)
        if self.settings.debug:
            p = pl.pos
            lines = ["POS %.1f %.1f %.1f" % (p.x, p.y, p.z), "VEL %.1f %.1f %.1f" % (pl.vel.x, pl.vel.y, pl.vel.z),
                     "STATE %s  GND %s" % (pl.state.upper(), "Y" if pl.grounded else "N"),
                     "FACES %d  CHUNKS %d/%d" % (self.r.stats_faces, self.r.stats_chunks, len(self.area.meshes)),
                     "NODE %s" % self.cur_node, "FLAGS %d  SECRETS %d" % (len(w.flags), len(w.secrets)),
                     "SEED %d  AREAS %d" % (w.seed, len(w.discovered))]
            for i, l in enumerate(lines):
                f.draw(s, l, 6, 78 + i * 9, (180, 255, 200))
        if self.dialog:
            n, i, lines = self.dialog
            box = pygame.Rect(8, RH - 64, RW - 16, 56)
            pygame.draw.rect(s, (10, 10, 30), box)
            pygame.draw.rect(s, (200, 200, 255), box, 1)
            f.draw(s, n.name, 14, RH - 60, (255, 220, 120))
            for k, ln in enumerate(f.wrap(lines[min(i, len(lines) - 1)], 66)[:4]):
                f.draw(s, ln, 14, RH - 49 + k * 9, (240, 240, 255))
            if int(self.t * 3) % 2 == 0:
                f.draw(s, "F >", RW - 16, RH - 16, (255, 255, 120), right=True)
        if self.player.inv > 1.2:
            self.overlay.fill((255, 0, 0, 60))
            s.blit(self.overlay, (0, 0))

    def draw_menu(self, title, items, values=None, y0=70, sub=None):
        s = self.r.surf
        f = self.font
        f.draw(s, title, RW // 2, y0 - 34, (255, 240, 160), scale=2, center=True)
        if sub:
            f.draw(s, sub, RW // 2, y0 - 16, (180, 180, 220), center=True)
        for i, it in enumerate(items):
            sel = i == self.menu_i
            txt = it
            if values:
                v = values(it)
                if v:
                    txt = "%s:  %s" % (it, v)
            col = (255, 255, 120) if sel else (210, 210, 230)
            if it == 'CONTINUE' and self.world is None:
                col = (110, 110, 130)
            if sel:
                txt = "> " + txt + " <"
            f.draw(s, txt, RW // 2, y0 + i * 12, col, center=True)

    def draw_title(self):
        s = self.r.surf
        if self.title_area:
            t = self.t
            cam = self.cam
            cam.target.set(0, 6, 20)
            cam.yaw = t * 0.07
            cam.pitch = 0.22
            cam.dist = 55
            cam._place(None)
            self.draw_world(self.title_area, cam, t, entities=False)
            cam.dist = 7.5
        else:
            s.fill((10, 10, 30))
        s.blit(self.dim, (0, 0))
        f = self.font
        wob = int(math.sin(self.t * 2) * 2)
        f.draw(s, "cat's b3313", RW // 2, 18 + wob, hsv(self.t * 0.05, 0.4, 1.0), scale=3, center=True)
        f.draw(s, "B3313 1.0 DECOMP  /  FILES = OFF", RW // 2, 46, (220, 220, 255), center=True)
        f.draw(s, "FILES = OFF", RW // 2, 56, (255, 180, 180), center=True)
        f.draw(s, "PYTHON 3.14 + PYGAME-CE", RW // 2, 66, (180, 220, 255), center=True)
        f.draw(s, "60 FPS", RW // 2, 76, (180, 255, 180), center=True)
        items = self.menu_items()
        for i, it in enumerate(items):
            sel = i == self.menu_i
            col = (255, 255, 120) if sel else (220, 220, 240)
            if it == 'CONTINUE' and self.world is None:
                col = (110, 110, 130)
            f.draw(s, ("> %s <" % it) if sel else it, RW // 2, 100 + i * 13, col, scale=1, center=True)
        note = "SESSION IN MEMORY: %d/%d POWER STARS" % (len(self.world.stars), self.world.total) if self.world else \
            "FILES = OFF - PROGRESS LIVES IN MEMORY FOR THIS SESSION"
        f.draw(s, note, RW // 2, RH - 30, (170, 170, 200), center=True)
        f.draw(s, "SEED %d   B3313 1.0 DECOMP" % self.settings.seed, RW // 2, RH - 18, (140, 140, 170),
               center=True)
        for m in self.msgs[-1:]:
            f.draw(s, m[0], RW // 2, RH - 44, (255, 200, 150), center=True)

    def draw_controls(self):
        s = self.r.surf
        s.fill((12, 10, 30))
        f = self.font
        f.draw(s, "CONTROLS", RW // 2, 10, (255, 240, 160), scale=2, center=True)
        lines = ["WASD / ARROWS ...... MOVE", "SPACE .............. JUMP (CHAIN 3 FOR A TRIPLE)",
                 "SHIFT .............. RUN", "Z / C / CTRL ....... CROUCH",
                 "CROUCH + JUMP ...... BACKFLIP (HIGH JUMP)", "RUN + CROUCH + JUMP  LONG JUMP",
                 "CROUCH IN AIR ...... GROUND POUND", "Q / E .............. ORBIT CAMERA",
                 "PGUP/PGDN, I/K ..... CAMERA PITCH", "RIGHT-DRAG MOUSE ... ORBIT CAMERA",
                 "M .................. TOGGLE MOUSE CAPTURE", "MOUSE WHEEL ........ CAMERA DISTANCE",
                 "F .................. TALK", "TAB ................ AREA MAP", "R .................. RESPAWN",
                 "ESC ................ PAUSE / MENU", "F2 / F3 ............ JITTER / DEBUG",
                 "STOMP ENEMIES. POUND THE SPIKY ONES. AVOID THE CUBES."]
        for i, l in enumerate(lines):
            f.draw(s, l, 40, 34 + i * 10, (220, 220, 240))
        f.draw(s, "ESC / ENTER: BACK", RW // 2, RH - 12, (150, 150, 190), center=True)

    def draw_about(self):
        s = self.r.surf
        s.fill((12, 10, 30))
        f = self.font
        f.draw(s, "ABOUT", RW // 2, 10, (255, 240, 160), scale=2, center=True)
        text = ("cat's b3313 IS A B3313 1.0 DECOMP-STYLE SOFTWARE-RENDERED 3D PLATFORMING ENGINE. IDENTIFIERS FOLLOW SM64 DECOMP LAYOUT (MARIOSTATE, LEVELSCRIPT, SURFACE, WARPNODE, SAVEFILE, GRAPHRENDERER). EVERY POLYGON, COLOUR, LETTER AND SOUND IS GENERATED FROM CODE AND MATHEMATICS AT RUNTIME. NO ROM ASSETS ARE LOADED. FILES = OFF: NOTHING IS WRITTEN TO DISK. PEACH'S CASTLE IS NONLINEAR IN THE B3313 1.0 SPIRIT: WARPS CHANGE THEIR MINDS, PAINTINGS LEAD TO PARALLEL LOBBIES, AND POWER STARS OPEN LOCKED DOORS. NOT AFFILIATED WITH NINTENDO. B3313 1.0 DECOMP.")
        for i, l in enumerate(f.wrap(text, 64)):
            f.draw(s, l, 20, 36 + i * 11, (220, 220, 240))
        f.draw(s, "ESC / ENTER: BACK", RW // 2, RH - 12, (150, 150, 190), center=True)

    def map_layout(self):
        w = self.world
        nodes = set(w.discovered)
        unknown = set()
        edges = set()
        for a, b in w.edges:
            edges.add((a, b, True))
        for n, outs in w.known_exits.items():
            for o in outs:
                if o not in w.discovered:
                    q = "?" + n + ">" + o
                    unknown.add(q)
                    edges.add((n, q, False))
                elif (n, o) not in w.edges and n != o:
                    edges.add((n, o, False))
        allnodes = list(nodes) + list(unknown)
        key = (len(allnodes), len(edges))
        if self.map_cache and self.map_cache[0] == key:
            return self.map_cache[1]
        pos = {}
        for n in allnodes:
            h = seed_hash(w.seed, n)
            a = (h % 3600) / 3600 * TAU
            rr = 20 + (h >> 12) % 80
            pos[n] = [math.cos(a) * rr, math.sin(a) * rr]
        adj = [(a, b) for a, b, _ in edges if a in pos and b in pos]
        for it in range(90):
            force = {n: [0.0, 0.0] for n in allnodes}
            for i, a in enumerate(allnodes):
                pa = pos[a]
                for b in allnodes[i + 1:]:
                    pb = pos[b]
                    dx, dy = pa[0] - pb[0], pa[1] - pb[1]
                    d2 = dx * dx + dy * dy + 0.01
                    f_ = 2600.0 / d2
                    d = math.sqrt(d2)
                    force[a][0] += dx / d * f_
                    force[a][1] += dy / d * f_
                    force[b][0] -= dx / d * f_
                    force[b][1] -= dy / d * f_
            for a, b in adj:
                pa, pb = pos[a], pos[b]
                dx, dy = pb[0] - pa[0], pb[1] - pa[1]
                d = math.hypot(dx, dy) + 0.01
                f_ = (d - 60) * 0.05
                force[a][0] += dx / d * f_
                force[a][1] += dy / d * f_
                force[b][0] -= dx / d * f_
                force[b][1] -= dy / d * f_
            for n in allnodes:
                fx, fy = force[n]
                fl = math.hypot(fx, fy)
                if fl > 6:
                    fx, fy = fx / fl * 6, fy / fl * 6
                pos[n][0] += fx - pos[n][0] * 0.004
                pos[n][1] += fy - pos[n][1] * 0.004
        res = (pos, edges, unknown)
        self.map_cache = (key, res)
        return res

    def draw_map(self):
        """Discovered-area map, drawn at window resolution so the growing network stays legible."""
        s = self.hires
        s.fill((8, 8, 22))
        f = self.font
        w = self.world
        pos, edges, unknown = self.map_layout()
        W, H = s.get_size()
        for i in range(0, W, 40):
            pygame.draw.line(s, (16, 16, 36), (i, 0), (i, H))
        for j in range(0, H, 40):
            pygame.draw.line(s, (16, 16, 36), (0, j), (W, j))
        if pos:
            xs = [p[0] for p in pos.values()]
            ys = [p[1] for p in pos.values()]
            minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
            sx = (W - 260) / max(1.0, maxx - minx)
            sy = (H - 170) / max(1.0, maxy - miny)
            sx, sy = min(sx, 6.0), min(sy, 6.0)
            ox = W / 2 - (minx + maxx) / 2 * sx
            oy = H / 2 + 20 - (miny + maxy) / 2 * sy
            P = lambda n: (int(ox + pos[n][0] * sx), int(oy + pos[n][1] * sy))
            for a, b, real in edges:
                if a in pos and b in pos:
                    pygame.draw.line(s, (110, 140, 255) if real else (60, 60, 95), P(a), P(b), 2 if real else 1)
            for n in pos:
                x, y = P(n)
                if n in unknown:
                    label, col = "VANISH BOO", (120, 120, 150)
                else:
                    label = w.node_names.get(n, n).replace("ENDLESS ", "").replace("UNDERGROUND ", "U. ")
                    label = label.replace("INTERIOR", "").replace("CORRIDORS", "").strip()[:18]
                    cur = n == self.cur_node
                    col = (255, 255, 120) if cur and int(self.t * 4) % 2 else (225, 235, 255)
                pygame.draw.rect(s, col, (x - 4, y - 4, 9, 9))
                f.draw(s, label, x, y + 7, col, scale=2, center=True)
        f.draw(s, "COURSE MAP  -  B3313 1.0 AS YOU REMEMBER IT", W // 2, 12, (255, 240, 160), scale=3, center=True)
        f.draw(s, "DISCOVERED %d   STARS %d/%d   SECRETS %d   PORTALS TAKEN %d   ??? = UNVISITED EXITS" %
               (len(w.discovered), len(w.stars), w.total, len(w.secrets), w.actions['portals']), W // 2, 42,
               (180, 180, 220), scale=2, center=True)
        f.draw(s, "ESC / TAB: BACK", W // 2, H - 22, (150, 150, 190), scale=2, center=True)

    def draw_loading(self):
        s = self.r.surf
        s.fill((6, 4, 16))
        f = self.font
        f.draw(s, "cat's b3313", RW // 2, 70, (255, 240, 160), scale=2, center=True)
        f.draw(s, "B3313 1.0 DECOMP  -  FILES = OFF  -  GENERATING FROM MATHEMATICS", RW // 2, 96, (180, 180, 220), center=True)
        pygame.draw.rect(s, (60, 60, 90), (80, 130, RW - 160, 10), 1)
        pygame.draw.rect(s, (140, 200, 255), (82, 132, int((RW - 164) * self.load_p), 6))
        f.draw(s, self.load_msg + "...", RW // 2, 150, (200, 200, 240), center=True)
        if not self.audio.ok:
            f.draw(s, "(AUDIO UNAVAILABLE - RUNNING SILENT)", RW // 2, 166, (255, 150, 150), center=True)

    def draw_ending(self):
        s = self.r.surf
        t = self.ending_t
        s.fill(lerpc((255, 255, 255), (20, 10, 40), min(1.0, t / 3)))
        f = self.font
        w = self.world
        lines = ["YOU FOUND THE LAST POWER STAR.", "", "THE CASTLE REMEMBERS YOU.", "",
                 "STARS: %d / %d" % (len(w.stars), w.total), "AREAS REMEMBERED: %d" % len(w.discovered),
                 "SECRETS: %d" % len(w.secrets), "PORTALS TAKEN: %d" % w.actions['portals'],
                 "JUMPS: %d   BACKFLIPS: %d   LONG JUMPS: %d" % (w.actions['jumps'], w.actions['backflips'],
                                                                 w.actions['longjumps']),
                 "COURSE TIME: %d:%02d" % (int(w.time) // 60, int(w.time) % 60), "",
                 "cat's b3313", "B3313 1.0 DECOMP", "FILES = OFF",
                 "", "PRESS ENTER TO RETURN TO THE CASTLE GROUNDS."]
        y = RH - t * 22
        for l in lines:
            if -10 < y < RH:
                f.draw(s, l, RW // 2, int(y), hsv(t * 0.05 + y * 0.002, 0.3, 1.0), center=True)
            y += 14
        if y < 0:
            f.draw(s, "PRESS ENTER", RW // 2, RH // 2, (255, 255, 200), center=True)

    def draw(self):
        st = self.state
        s = self.r.surf
        if st == 'loading':
            self.draw_loading()
        elif st == 'title':
            self.draw_title()
        elif st in ('play', 'fade_out', 'fade_in', 'pause', 'gameover') or \
                (st in ('map', 'controls', 'settings', 'about') and self.prev_state in ('pause', 'play')
                 and st != 'map' and st != 'controls' and st != 'about'):
            self.draw_world(self.area, self.cam, self.t)
            self.draw_hud()
            if st in ('fade_out', 'fade_in'):
                self.overlay.fill((255, 255, 255, int(255 * clamp(self.fade, 0, 1))))
                s.blit(self.overlay, (0, 0))
            if st == 'pause':
                s.blit(self.dim, (0, 0))
                self.draw_menu("PAUSED", self.menu_items(), y0=78,
                               sub="%s  -  %d/%d STARS" % (self.area.name, len(self.world.stars), self.world.total))
            if st == 'settings':
                s.blit(self.dim, (0, 0))
                self.draw_menu("SETTINGS", self.menu_items(), self.setting_value, y0=62)
            if st == 'gameover':
                s.blit(self.dim, (0, 0))
                self.font.draw(s, "GAME OVER...", RW // 2, 90, (255, 120, 120), scale=2, center=True)
                self.font.draw(s, "PRESS ENTER TO WAKE UP IN THE CASTLE", RW // 2, 124, (230, 230, 255), center=True)
                self.font.draw(s, "(YOUR POWER STARS ARE STILL IN MEMORY)", RW // 2, 136, (170, 170, 200), center=True)
        elif st == 'settings':
            if self.title_area:
                self.draw_title()
            s.blit(self.dim, (0, 0))
            s.blit(self.dim, (0, 0))
            self.draw_menu("SETTINGS", self.menu_items(), self.setting_value, y0=62,
                           sub="LEFT/RIGHT TO CHANGE")
        elif st == 'map':
            self.draw_map()
            self.screen.blit(self.hires, (0, 0))
            pygame.display.flip()
            return
        elif st == 'controls':
            self.draw_controls()
        elif st == 'about':
            self.draw_about()
        elif st == 'ending':
            self.draw_ending()
        ox = oy = 0
        if self.shake > 0:
            ox = random.randint(-3, 3)
            oy = random.randint(-3, 3)
        self.screen.fill((0, 0, 0))
        big = pygame.transform.scale(s, self.scaled_size)
        self.screen.blit(big, (self.blit_off[0] + ox, self.blit_off[1] + oy))
        pygame.display.flip()

    # ------------------------------------------------------------------ main loop
    def run(self):
        while self.running:
            cap = self.settings.fps if self.state != 'loading' else 0
            dt = self.clock.tick(cap) / 1000.0
            dt = min(dt, 0.1)
            if self.state in ('title',):
                self.t += dt
            for ev in pygame.event.get():
                self.handle_event(ev)
            self.update(dt)
            self.draw()
        pygame.quit()


def main():
    try:
        g = B3313Engine()
        g.run()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            pygame.quit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
