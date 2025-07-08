---
title: "How I 'made' my custom media suite"
date: "2025-08-07"
theme: "Coding"
summary: "Explanation of the differents bricks used to make a media suite, for me and my family. Using open source docker projects."
image: "https://github.com/Epwo/articles/blob/d9b4ed48e8e9c0a4c082673b9595e06ec3ebe2b2/images/stream_stack/stream_stack.png"
---

# The purpose
The goal of this was to learn more about docker and devOps in general, like how to deploy stuff ? how to montior thoses ? How does the domain name stuff actually works ?

# The hardware
Because I like doing stuff myself, and not to rely on some third-party ( and mostly to avoid paying ). I chose to try and host my own server at my home, so I took my old laptop from high school. Installed debian on it, and call it a go.
Turns out, this old laptop was really too old. So i went ahead and bought an old (but powerful enough) computer on the marketplace. This one is powerful enough, and is the one currently in use till now!

# The stack
## Disclamer
This is obviously for movies, my and my family own physically ( via blue-rays or CDs ). Downloading movies you don't own is illegal (I think?)

## Schematic
![the schematic](../images/stream_stack/schematic.svg)


## How it works ?

Every brick in the stack is talking with the next one, via simple REST API calls. (which are obviously secured, and needs auth tokens )

*Work in progress*
