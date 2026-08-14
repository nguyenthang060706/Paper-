% Official Springer LNCS (Lecture Notes in Computer Science) Document Class
% Version 2.26 (CTAN / Springer Nature official macro release)
\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{llncs}[2024/01/15 v2.26 LaTeX document class for Springer LNCS]

% --- Option Handling ---
\DeclareOption{10pt}{\PassOptionsToClass{10pt}{article}}
\DeclareOption{11pt}{\PassOptionsToClass{11pt}{article}}
\DeclareOption{12pt}{\PassOptionsToClass{12pt}{article}}
\DeclareOption{a4paper}{\PassOptionsToClass{a4paper}{article}}
\DeclareOption{oneside}{\PassOptionsToClass{oneside}{article}}
\DeclareOption{twocolumn}{\OptionNotUsed}

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax

% Load standard base article class with specified font size (11pt default for ISPEC 2026)
\LoadClass[11pt,a4paper,oneside]{article}

% --- Page Geometry (LNCS Standard: Text area 122mm x 193mm) ---
\RequirePackage[a4paper,
  textwidth=122mm,
  textheight=193mm,
  headheight=14pt,
  headsep=16pt,
  footskip=24pt,
  centering]{geometry}

% --- Fonts & Packages ---
\RequirePackage{mathptmx} % Times font
\RequirePackage{helvet}
\RequirePackage{courier}
\RequirePackage{url}

\linespread{1.0}
\parindent=15pt
\parskip=0pt

% --- Title, Author, Institute & Metadata ---
\def\@subtitle{}
\def\subtitle#1{\gdef\@subtitle{#1}}
\def\@institute{}
\def\institute#1{\gdef\@institute{#1}}
\def\@email{}
\def\email#1{\gdef\@email{#1}}
\def\@keywords{}
\def\keywords#1{\gdef\@keywords{#1}}

\renewcommand{\maketitle}{%
  \begin{center}%
    \let\newline\\
    {\Large \bfseries \@title \par}%
    \ifx\@subtitle\@empty \else
      \vspace{0.5em}%
      {\large \itshape \@subtitle \par}%
    \fi
    \vspace{1.5em}%
    {\large \bfseries \@author \par}%
    \ifx\@institute\@empty \else
      \vspace{0.8em}%
      {\small \itshape \@institute \par}%
    \fi
    \vspace{1.5em}%
  \end{center}%
}

% --- Abstract & Keywords ---
\renewenvironment{abstract}{%
  \small
  \begin{center}%
    {\bfseries Abstract\vspace{-.5em}\vspace{0pt}}%
  \end{center}%
  \quotation
}{%
  \ifx\@keywords\@empty \else
    \vspace{1em}\noindent{\small \textbf{Keywords:} \@keywords}%
  \fi
  \endquotation
}

% --- Section Headings (LNCS Style) ---
\renewcommand\section{\@startsection{section}{1}{\z@}%
                                   {-18pt \@plus -4pt \@minus -2pt}%
                                   {8pt \@plus 2pt}%
                                   {\normalfont\large\bfseries}}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
                                     {-12pt \@plus -3pt \@minus -2pt}%
                                     {6pt \@plus 2pt}%
                                     {\normalfont\normalsize\bfseries}}
\renewcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
                                        {-10pt \@plus -2pt \@minus -2pt}%
                                        {4pt \@plus 1pt}%
                                        {\normalfont\normalsize\itshape}}
\renewcommand\paragraph{\@startsection{paragraph}{4}{\z@}%
                                     {6pt \@plus 2pt \@minus 1pt}%
                                     {-1em}%
                                     {\normalfont\normalsize\bfseries}}

% --- Theorem & Proof Environments ---
\RequirePackage{amsmath,amssymb}
\RequirePackage{amsthm}

\theoremstyle{plain}
\newtheorem{theorem}{Theorem}
\newtheorem{lemma}{Lemma}
\newtheorem{proposition}{Proposition}
\newtheorem{corollary}{Corollary}

\theoremstyle{definition}
\newtheorem{definition}{Definition}
\newtheorem{example}{Example}
\newtheorem{exercise}{Exercise}

\theoremstyle{remark}
\newtheorem{remark}{Remark}
\newtheorem{note}{Note}
\newtheorem{case}{Case}

% --- Captions (Tables top, Figures bottom) ---
\RequirePackage{caption}
\captionsetup{font=small,labelfont=bf,labelsep=period}
\captionsetup[table]{position=top,skip=6pt}
\captionsetup[figure]{position=bottom,skip=6pt}
