r"""Generate calculus-equation training images with Matplotlib MathText.

Default usage:

    .\.venv\Scripts\python.exe utilities\calculus_equation_generator.py

The command creates 180 unique RGB 224 x 224 PNG files in
``data_calculus_equations\generated``. Formula captions and class labels are
not included in the images.
"""

from __future__ import annotations

import argparse
import os
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

MATPLOTLIB_CONFIG_DIR = Path(__file__).resolve().parents[1] / ".matplotlib"
MATPLOTLIB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CONFIG_DIR))

import matplotlib as mpl
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageFilter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data_calculus_equations" / "generated"
DEFAULT_COUNT = 180
DEFAULT_SIZE = 224
DEFAULT_SEED = 20260729


@dataclass(frozen=True)
class FormulaSpec:
    category: str
    latex: str


@dataclass(frozen=True)
class FormulaStyle:
    name: str
    background: tuple[int, int, int]
    foreground: tuple[int, int, int]
    fontset: str
    paper: bool = False
    rotation: float = 0.0
    blur: float = 0.0


STYLES: tuple[FormulaStyle, ...] = (
    FormulaStyle(
        name="textbook",
        background=(255, 255, 253),
        foreground=(24, 27, 32),
        fontset="stix",
        rotation=0.4,
    ),
    FormulaStyle(
        name="dark",
        background=(17, 24, 34),
        foreground=(238, 244, 251),
        fontset="dejavusans",
        rotation=0.7,
    ),
    FormulaStyle(
        name="notebook",
        background=(249, 247, 235),
        foreground=(31, 75, 142),
        fontset="cm",
        paper=True,
        rotation=3.0,
        blur=0.12,
    ),
)


def _formula_catalog() -> tuple[FormulaSpec, ...]:
    formulas: dict[str, tuple[str, ...]] = {
        "limits": (
            r"\lim_{x\to 0}\frac{\sin x}{x}=1",
            r"\lim_{x\to 0}\frac{1-\cos x}{x^2}=\frac{1}{2}",
            r"\lim_{x\to 0}\frac{e^x-1}{x}=1",
            r"\lim_{x\to 0}\frac{\ln(1+x)}{x}=1",
            r"\lim_{x\to 0}(1+x)^{1/x}=e",
            r"\lim_{x\to\infty}\left(1+\frac{1}{x}\right)^x=e",
            r"\lim_{x\to\infty}\frac{\ln x}{x}=0",
            r"\lim_{x\to\infty}\frac{x^2}{e^x}=0",
            r"\lim_{x\to a}\frac{x^2-a^2}{x-a}=2a",
            r"\lim_{h\to 0}\frac{(x+h)^n-x^n}{h}=nx^{n-1}",
            r"\lim_{x\to 0}\frac{\tan x}{x}=1",
            r"\lim_{x\to\infty}\left(\sqrt{x^2+x}-x\right)=\frac{1}{2}",
        ),
        "derivative-rules": (
            r"f'(x)=\lim_{h\to 0}\frac{f(x+h)-f(x)}{h}",
            r"\frac{d}{dx}x^n=nx^{n-1}",
            r"\frac{d}{dx}[f(x)g(x)]=f'g+fg'",
            r"\frac{d}{dx}\frac{f}{g}=\frac{f'g-fg'}{g^2}",
            r"\frac{d}{dx}f(g(x))=f'(g(x))g'(x)",
            r"\frac{d}{dx}e^{kx}=ke^{kx}",
            r"\frac{d}{dx}\ln x=\frac{1}{x}",
            r"\frac{d}{dx}\sin x=\cos x",
            r"\frac{d}{dx}\cos x=-\sin x",
            r"\frac{d}{dx}\arctan x=\frac{1}{1+x^2}",
            r"\frac{d}{dx}a^x=a^x\ln a",
            r"\frac{d}{dx}\sinh x=\cosh x",
        ),
        "derivative-examples": (
            r"\frac{d}{dx}(x^3-4x+1)=3x^2-4",
            r"\frac{d}{dx}e^{x^2}=2xe^{x^2}",
            r"\frac{d}{dx}\ln(\sin x)=\frac{\cos x}{\sin x}",
            r"\frac{d}{dx}x^x=x^x(1+\ln x)",
            r"\frac{d}{dx}\sin(x^2)=2x\cos(x^2)",
            r"\frac{d}{dx}\sqrt{1+x^2}=\frac{x}{\sqrt{1+x^2}}",
            r"\frac{d}{dx}\frac{x}{1+x^2}=\frac{1-x^2}{(1+x^2)^2}",
            r"\frac{d}{dx}\tan x=\frac{1}{\cos^2x}",
            r"\frac{d}{dx}\arcsin x=\frac{1}{\sqrt{1-x^2}}",
            r"x^2+y^2=1\quad\Rightarrow\quad\frac{dy}{dx}=-\frac{x}{y}",
            r"\frac{d}{dx}(x^2\ln x)=2x\ln x+x",
            r"\frac{d}{dx}(e^x\cos x)=e^x(\cos x-\sin x)",
        ),
        "indefinite-integrals": (
            r"\int x^n\,dx=\frac{x^{n+1}}{n+1}+C",
            r"\int\frac{1}{x}\,dx=\ln|x|+C",
            r"\int e^{kx}\,dx=\frac{1}{k}e^{kx}+C",
            r"\int\sin x\,dx=-\cos x+C",
            r"\int\cos x\,dx=\sin x+C",
            r"\int\frac{1}{\cos^2x}\,dx=\tan x+C",
            r"\int\frac{dx}{1+x^2}=\arctan x+C",
            r"\int\frac{dx}{\sqrt{1-x^2}}=\arcsin x+C",
            r"\int\ln x\,dx=x\ln x-x+C",
            r"\int xe^{x^2}\,dx=\frac{1}{2}e^{x^2}+C",
            r"\int x\ln x\,dx=\frac{x^2}{2}\ln x-\frac{x^2}{4}+C",
            r"\int\frac{dx}{a^2+x^2}=\frac{1}{a}\arctan\frac{x}{a}+C",
        ),
        "definite-integrals": (
            r"\int_0^\pi\sin x\,dx=2",
            r"\int_0^{\pi/2}\cos x\,dx=1",
            r"\int_0^1x^p\,dx=\frac{1}{p+1}",
            r"\int_0^\infty e^{-x}\,dx=1",
            r"\int_{-\infty}^{\infty}e^{-x^2}\,dx=\sqrt{\pi}",
            r"\int_0^1\ln x\,dx=-1",
            r"\int_0^1\frac{dx}{1+x^2}=\frac{\pi}{4}",
            r"\int_0^{2\pi}\sin^2x\,dx=\pi",
            r"\int_0^a x\,dx=\frac{a^2}{2}",
            r"\int_1^e\frac{dx}{x}=1",
            r"\int_0^1x(1-x)\,dx=\frac{1}{6}",
            r"\int_{-1}^{1}x^2\,dx=\frac{2}{3}",
        ),
        "integration-methods": (
            r"\int u\,dv=uv-\int v\,du",
            r"\int f(g(x))g'(x)\,dx=\int f(u)\,du",
            r"\int x\cos x\,dx=x\sin x+\cos x+C",
            r"\int x^2e^x\,dx=e^x(x^2-2x+2)+C",
            r"\frac{1}{x^2-a^2}=\frac{1}{2a}\left(\frac{1}{x-a}-\frac{1}{x+a}\right)",
            r"\int\sin^2x\,dx=\frac{x}{2}-\frac{\sin2x}{4}+C",
            r"\int\cos^2x\,dx=\frac{x}{2}+\frac{\sin2x}{4}+C",
            r"\int\frac{dx}{\sqrt{x^2+a^2}}=\ln|x+\sqrt{x^2+a^2}|+C",
            r"\int\frac{dx}{x^2-a^2}=\frac{1}{2a}\ln\left|\frac{x-a}{x+a}\right|+C",
            r"\int_0^1f(x)\,dx=\int_0^1f(1-x)\,dx",
            r"\int e^x\sin x\,dx=\frac{e^x}{2}(\sin x-\cos x)+C",
            r"u=x^2+1,\quad du=2x\,dx",
        ),
        "series": (
            r"\sum_{n=0}^{\infty}ar^n=\frac{a}{1-r}",
            r"\sum_{n=1}^{\infty}\frac{1}{n^2}=\frac{\pi^2}{6}",
            r"\sum_{n=1}^{\infty}\frac{(-1)^{n+1}}{n}=\ln2",
            r"\sum_{n=0}^{\infty}x^n=\frac{1}{1-x}",
            r"\sum_{n=1}^{\infty}nx^{n-1}=\frac{1}{(1-x)^2}",
            r"\sum_{n=1}^{\infty}\frac{x^n}{n}=-\ln(1-x)",
            r"L=\lim_{n\to\infty}\left|\frac{a_{n+1}}{a_n}\right|<1",
            r"L=\lim_{n\to\infty}\sqrt[n]{|a_n|}<1",
            r"\sum_{n=1}^{\infty}\frac{1}{n^p}<\infty\quad(p>1)",
            r"\frac{1}{R}=\lim_{n\to\infty}\sqrt[n]{|c_n|}",
            r"\sum_{n=1}^{\infty}\frac{1}{2^n}=1",
            r"\sum_{n=1}^{\infty}\frac{n}{2^n}=2",
        ),
        "taylor-series": (
            r"e^x=\sum_{n=0}^{\infty}\frac{x^n}{n!}",
            r"\sin x=\sum_{n=0}^{\infty}\frac{(-1)^nx^{2n+1}}{(2n+1)!}",
            r"\cos x=\sum_{n=0}^{\infty}\frac{(-1)^nx^{2n}}{(2n)!}",
            r"\ln(1+x)=\sum_{n=1}^{\infty}\frac{(-1)^{n+1}x^n}{n}",
            r"\frac{1}{1-x}=\sum_{n=0}^{\infty}x^n",
            r"\arctan x=\sum_{n=0}^{\infty}\frac{(-1)^nx^{2n+1}}{2n+1}",
            r"\sinh x=\sum_{n=0}^{\infty}\frac{x^{2n+1}}{(2n+1)!}",
            r"\cosh x=\sum_{n=0}^{\infty}\frac{x^{2n}}{(2n)!}",
            r"f(x)=\sum_{n=0}^{\infty}\frac{f^{(n)}(a)}{n!}(x-a)^n",
            r"(1+x)^\alpha=1+\alpha x+\frac{\alpha(\alpha-1)}{2!}x^2+\cdots",
            r"\sqrt{1+x}=1+\frac{x}{2}-\frac{x^2}{8}+\cdots",
            r"e^{-x}=\sum_{n=0}^{\infty}\frac{(-1)^nx^n}{n!}",
        ),
        "differential-equations": (
            r"\frac{dy}{dx}=ky\quad\Rightarrow\quad y=Ce^{kx}",
            r"\frac{dy}{dx}=xy\quad\Rightarrow\quad y=Ce^{x^2/2}",
            r"y'+p(x)y=q(x)",
            r"y''+\omega^2y=0\quad\Rightarrow\quad y=A\cos\omega x+B\sin\omega x",
            r"y''-\lambda^2y=0\quad\Rightarrow\quad y=Ae^{\lambda x}+Be^{-\lambda x}",
            r"\frac{dy}{dt}=ry\left(1-\frac{y}{K}\right)",
            r"\frac{dy}{g(y)}=f(x)\,dx",
            r"M(x,y)\,dx+N(x,y)\,dy=0",
            r"x^2y''+axy'+by=0",
            r"y'=2x,\ y(0)=1\quad\Rightarrow\quad y=x^2+1",
            r"y'+y=e^x\quad\Rightarrow\quad y=\frac{1}{2}e^x+Ce^{-x}",
            r"y''+y=0,\ y(0)=0,\ y'(0)=1\quad\Rightarrow\quad y=\sin x",
        ),
        "partial-derivatives": (
            r"f_x(a,b)=\lim_{h\to0}\frac{f(a+h,b)-f(a,b)}{h}",
            r"\nabla f=\left(\frac{\partial f}{\partial x},\frac{\partial f}{\partial y}\right)",
            r"\frac{\partial}{\partial x}(x^2y)=2xy",
            r"\frac{\partial}{\partial y}e^{xy}=xe^{xy}",
            r"\frac{\partial^2f}{\partial x\partial y}=\frac{\partial^2f}{\partial y\partial x}",
            r"D_{\mathbf{u}}f=\nabla f\cdot\mathbf{u}",
            r"df=f_x\,dx+f_y\,dy",
            r"\frac{\partial z}{\partial t}=z_x\frac{dx}{dt}+z_y\frac{dy}{dt}",
            r"\nabla^2f=f_{xx}+f_{yy}",
            r"z=f(a,b)+f_x(a,b)(x-a)+f_y(a,b)(y-b)",
            r"\frac{\partial}{\partial x}e^{xy}=ye^{xy}",
            r"H_f=\left(f_{ij}\right)_{2\times2}",
        ),
        "multiple-integrals": (
            r"\iint_R f(x,y)\,dA=\int_a^b\int_c^d f(x,y)\,dy\,dx",
            r"\iint_D f\,dA=\int_{\alpha}^{\beta}\int_0^{g(\theta)}f(r,\theta)r\,dr\,d\theta",
            r"\iiint_E f\,dV=\int\int\int f(x,y,z)\,dz\,dy\,dx",
            r"\iint_D f(x,y)\,dA=\iint_S f(x(u,v),y(u,v))|J|\,du\,dv",
            r"A(D)=\iint_D1\,dA",
            r"V=\iint_D f(x,y)\,dA",
            r"\bar{x}=\frac{1}{M}\iint_Dx\rho(x,y)\,dA",
            r"\int_Cf\,ds=\int_a^bf(\mathbf{r}(t))|\mathbf{r}'(t)|\,dt",
            r"\iint_Sf\,dS=\iint_Df(\mathbf{r}(u,v))|\mathbf{r}_u\times\mathbf{r}_v|\,du\,dv",
            r"\iiint_E1\,dV=\int_0^{2\pi}\int_0^\pi\int_0^Rr^2\sin\phi\,dr\,d\phi\,d\theta",
            r"dA=r\,dr\,d\theta",
            r"dV=r\,dr\,d\theta\,dz",
        ),
        "vector-calculus": (
            r"\nabla f=f_x\mathbf{i}+f_y\mathbf{j}+f_z\mathbf{k}",
            r"\nabla\cdot\mathbf{F}=\frac{\partial P}{\partial x}+\frac{\partial Q}{\partial y}+\frac{\partial R}{\partial z}",
            r"\nabla\times\mathbf{F}=\left(R_y-Q_z,\ P_z-R_x,\ Q_x-P_y\right)",
            r"\int_C\mathbf{F}\cdot d\mathbf{r}=\int_a^b\mathbf{F}(\mathbf{r}(t))\cdot\mathbf{r}'(t)\,dt",
            r"\oint_C(P\,dx+Q\,dy)=\iint_D(Q_x-P_y)\,dA",
            r"\oint_C\mathbf{F}\cdot d\mathbf{r}=\iint_S(\nabla\times\mathbf{F})\cdot\mathbf{n}\,dS",
            r"\iint_S\mathbf{F}\cdot\mathbf{n}\,dS=\iiint_E\nabla\cdot\mathbf{F}\,dV",
            r"\mathbf{F}=\nabla f\quad\Rightarrow\quad\int_C\mathbf{F}\cdot d\mathbf{r}=f(B)-f(A)",
            r"\int_a^b\nabla f(\mathbf{r}(t))\cdot\mathbf{r}'(t)\,dt=f(\mathbf{r}(b))-f(\mathbf{r}(a))",
            r"\nabla^2f=f_{xx}+f_{yy}+f_{zz}",
            r"\nabla\cdot(\nabla\times\mathbf{F})=0",
            r"\nabla\times(\nabla f)=\mathbf{0}",
        ),
        "parametric-polar": (
            r"x=x(t),\quad y=y(t)",
            r"\frac{dy}{dx}=\frac{dy/dt}{dx/dt}",
            r"\frac{d^2y}{dx^2}=\frac{\frac{d}{dt}(dy/dx)}{dx/dt}",
            r"L=\int_a^b\sqrt{\left(\frac{dx}{dt}\right)^2+\left(\frac{dy}{dt}\right)^2}\,dt",
            r"x=r\cos\theta,\quad y=r\sin\theta",
            r"\frac{dy}{dx}=\frac{r'\sin\theta+r\cos\theta}{r'\cos\theta-r\sin\theta}",
            r"A=\frac{1}{2}\int_{\alpha}^{\beta}r^2\,d\theta",
            r"L=\int_{\alpha}^{\beta}\sqrt{r^2+\left(\frac{dr}{d\theta}\right)^2}\,d\theta",
            r"\kappa=\frac{|x'y''-y'x''|}{(x'^2+y'^2)^{3/2}}",
            r"x=a\cos t,\quad y=a\sin t",
            r"\frac{dx}{dt}=-a\sin t,\quad\frac{dy}{dt}=a\cos t",
            r"ds=\sqrt{(dx)^2+(dy)^2}",
        ),
        "applications": (
            r"L=\int_a^b\sqrt{1+[f'(x)]^2}\,dx",
            r"S=2\pi\int_a^bf(x)\sqrt{1+[f'(x)]^2}\,dx",
            r"V=\pi\int_a^b[f(x)]^2\,dx",
            r"V=\pi\int_a^b(R(x)^2-r(x)^2)\,dx",
            r"V=2\pi\int_a^bx f(x)\,dx",
            r"f_{\mathrm{avg}}=\frac{1}{b-a}\int_a^bf(x)\,dx",
            r"f'(c)=\frac{f(b)-f(a)}{b-a}",
            r"f(a)=f(b)\quad\Rightarrow\quad f'(c)=0",
            r"x_{n+1}=x_n-\frac{f(x_n)}{f'(x_n)}",
            r"f'(c)=0,\quad f''(c)>0",
            r"A=\int_a^b[f(x)-g(x)]\,dx",
            r"L(x)=f(a)+f'(a)(x-a)",
        ),
        "fundamental-results": (
            r"\frac{d}{dx}\int_a^xf(t)\,dt=f(x)",
            r"\int_a^bf'(x)\,dx=f(b)-f(a)",
            r"F(b)-F(a)=\int_a^bf'(x)\,dx",
            r"\frac{d}{dx}\int_{u(x)}^{v(x)}f(t)\,dt=f(v)v'-f(u)u'",
            r"\lim_{x\to a}\frac{f(x)}{g(x)}=\lim_{x\to a}\frac{f'(x)}{g'(x)}",
            r"R_n(x)=\frac{f^{(n+1)}(\xi)}{(n+1)!}(x-a)^{n+1}",
            r"\Gamma(s)=\int_0^\infty x^{s-1}e^{-x}\,dx",
            r"B(p,q)=\int_0^1x^{p-1}(1-x)^{q-1}\,dx",
            r"\Gamma(s+1)=s\Gamma(s)",
            r"\frac{d}{dx}\int_a^x\frac{dt}{t}=\frac{1}{x}",
            r"\frac{d}{dx}\int_0^{g(x)}f(t)\,dt=f(g(x))g'(x)",
            r"\int_a^bf(g(x))g'(x)\,dx=F(g(b))-F(g(a))",
        ),
    }

    catalog = tuple(
        FormulaSpec(category=category, latex=latex)
        for category, category_formulas in formulas.items()
        for latex in category_formulas
    )
    if len(catalog) != 180:
        raise RuntimeError(f"Expected 180 formulas, built {len(catalog)}")
    if len({spec.latex for spec in catalog}) != len(catalog):
        raise RuntimeError("Formula strings must be unique")
    return catalog


def _color_hex(color: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*color)


def _render_mathtext(
    latex: str,
    *,
    color: tuple[int, int, int],
    fontset: str,
) -> Image.Image:
    """Render one formula to a tightly cropped transparent RGBA image."""
    figure = Figure(figsize=(9, 3.2), dpi=100)
    figure.patch.set_alpha(0)
    canvas = FigureCanvasAgg(figure)
    axis = figure.add_axes((0, 0, 1, 1))
    axis.set_axis_off()
    axis.patch.set_alpha(0)

    font_size = 46 if len(latex) < 60 else 40
    with mpl.rc_context(
        {
            "mathtext.fontset": fontset,
            "font.family": "serif",
            "text.antialiased": True,
        }
    ):
        axis.text(
            0.5,
            0.5,
            f"${latex}$",
            color=_color_hex(color),
            fontsize=font_size,
            horizontalalignment="center",
            verticalalignment="center",
        )
        canvas.draw()

    rgba = np.asarray(canvas.buffer_rgba()).copy()
    rendered = Image.fromarray(rgba, mode="RGBA")
    alpha_box = rendered.getchannel("A").getbbox()
    if alpha_box is None:
        raise RuntimeError(f"Formula rendered empty: {latex}")

    padding = 10
    left, top, right, bottom = alpha_box
    crop_box = (
        max(0, left - padding),
        max(0, top - padding),
        min(rendered.width, right + padding),
        min(rendered.height, bottom + padding),
    )
    return rendered.crop(crop_box)


def _resize_to_fit(
    image: Image.Image,
    *,
    maximum_width: int,
    maximum_height: int,
) -> Image.Image:
    scale = min(
        maximum_width / image.width,
        maximum_height / image.height,
    )
    width = max(1, round(image.width * scale))
    height = max(1, round(image.height * scale))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _add_notebook_background(
    image: Image.Image,
    *,
    rng: random.Random,
) -> None:
    draw = ImageDraw.Draw(image)
    line_color = (217, 226, 229)
    line_spacing = rng.randint(25, 31)
    offset = rng.randint(4, line_spacing)
    for y in range(offset, image.height, line_spacing):
        draw.line((0, y, image.width, y), fill=line_color, width=1)

    if rng.random() < 0.55:
        margin_x = rng.randint(22, 35)
        draw.line(
            (margin_x, 0, margin_x, image.height),
            fill=(236, 195, 190),
            width=1,
        )

    for _ in range(70):
        x = rng.randrange(image.width)
        y = rng.randrange(image.height)
        delta = rng.choice((-1, 1)) * rng.randint(1, 5)
        base = image.getpixel((x, y))
        color = tuple(
            min(255, max(0, channel + delta)) for channel in base
        )
        image.putpixel((x, y), color)


def render_formula(
    formula: FormulaSpec,
    *,
    style: FormulaStyle,
    seed: int,
    size: int = DEFAULT_SIZE,
) -> Image.Image:
    """Render a formula as an RGB square image."""
    if size < 64:
        raise ValueError("size must be at least 64")

    rng = random.Random(seed)
    foreground = style.foreground
    if style.name == "notebook":
        foreground = rng.choice(
            ((28, 68, 132), (37, 72, 127), (57, 58, 61))
        )

    equation = _render_mathtext(
        formula.latex,
        color=foreground,
        fontset=style.fontset,
    )
    if style.rotation:
        angle = rng.uniform(-style.rotation, style.rotation)
        equation = equation.rotate(
            angle,
            resample=Image.Resampling.BICUBIC,
            expand=True,
        )

    equation = _resize_to_fit(
        equation,
        maximum_width=size - 20,
        maximum_height=round(size * 0.68),
    )
    canvas = Image.new("RGB", (size, size), style.background)
    if style.paper:
        _add_notebook_background(canvas, rng=rng)

    offset_x = rng.randint(-4, 4) if style.paper else rng.randint(-2, 2)
    offset_y = rng.randint(-5, 5) if style.paper else rng.randint(-3, 3)
    left = (size - equation.width) // 2 + offset_x
    top = (size - equation.height) // 2 + offset_y
    canvas.paste(equation, (left, top), equation)

    if style.blur:
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=style.blur))
    return canvas.convert("RGB")


def generate_dataset(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    count: int = DEFAULT_COUNT,
    size: int = DEFAULT_SIZE,
    seed: int = DEFAULT_SEED,
) -> list[Path]:
    """Generate unique calculus formula images and return their paths."""
    catalog = _formula_catalog()
    if not 1 <= count <= len(catalog):
        raise ValueError(f"count must be between 1 and {len(catalog)}")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for old_path in destination.glob("calculus-equation-*.png"):
        old_path.unlink()

    rng = random.Random(seed)
    output_paths: list[Path] = []
    selected_styles: list[str] = []

    for index, formula in enumerate(catalog[:count], start=1):
        style = rng.choice(STYLES)
        image_seed = rng.randrange(2**32)
        image = render_formula(
            formula,
            style=style,
            seed=image_seed,
            size=size,
        )
        output_name = (
            f"calculus-equation-{index:03d}-"
            f"{formula.category}-{style.name}.png"
        )
        output_path = destination / output_name
        image.save(output_path, format="PNG", optimize=True)
        output_paths.append(output_path)
        selected_styles.append(style.name)
        print(f"[{index}/{count}] Saved {output_path.name}")

    print(f"Styles: {dict(Counter(selected_styles))}")
    return output_paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate calculus equation images with Matplotlib MathText."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help=f"number of unique formulas, at most {DEFAULT_COUNT}",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_SIZE,
        help=f"output width and height (default: {DEFAULT_SIZE})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"random seed (default: {DEFAULT_SEED})",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = generate_dataset(
        args.output_dir,
        count=args.count,
        size=args.size,
        seed=args.seed,
    )
    print(
        f"Generated {len(paths)} calculus equation images in "
        f"{args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
