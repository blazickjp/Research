---
layout: post
title:  "How to set up MathJax on Jekyll and GitHub properly?"
date:   2017-03-28 10:44:32 +0100
categories: mypost
---
# Introduction

Yesterday I wrote about how to reference equations correctly with Microsoft Office 2013 and 2016. Today I'll write about how to set up MathJax, a beautiful equation rendering engine written in JavaScript, which handles LaTeX and MathML (and ASCIIMathML) to your Jekyll site on GitHub (and everywhere else on the Web of course). I found a handful of descriptions about it ([this][jekyll], [this][gastonsanchez] and [this][tobanwiebe] - [this][haixing-hu] could also be helpful), but those are not working for me. At the end, I got help from [here][pages-gem]. Thanks again, **hugomilan**! :)

# How to do it?
The recipe is simple.

* Go to your \_includes/head.html in your Jekyll folder structure.

* Put the following code snippet there (before the </head> tag).
    {% highlight html %}
       <head>
       ...
       <script type="text/x-mathjax-config"> MathJax.Hub.Config({ TeX: { equationNumbers: { autoNumber: "all" } } }); </script>
       <script type="text/x-mathjax-config">
         MathJax.Hub.Config({
           tex2jax: {
             inlineMath: [ ['$','$'], ["\\(","\\)"] ],
             processEscapes: true
           }
         });
       </script>
       <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
       </head>
    {% endhighlight %}

* And you are done! These three lines will do three things. It will allow you to use equations through the entire site, everywhere you want. The second thing is, that you can use equations with auto numbering using two dollar signs. For example, this equation <code>$$ E = m\cdot c^2 \label{eq:mc2}$$</code> will look like the equation below and you can refer to it as <code>\ref{eq:mc2}</code> (which will render to this: \ref{eq:mc2}).

    $$ E = M\cdot c^2 \label{eq:mc2} $$

    But you can use inline equations too (this is the third thing), with one dollar sign, like this: <code>$ J(x) = \int{L(t, x, \dot{x}) dt} \$</code>. The equation above will render to this: $ J(x) = \int{L(t, x, \dot{x}) dt} $.

That's all folks! (For today.)

[jekyll]: https://jekyllrb.com/docs/extras/
[gastonsanchez]: http://gastonsanchez.com/visually-enforced/opinion/2014/02/16/Mathjax-with-jekyll/
[tobanwiebe]: http://tobanwiebe.com/blog/2016/02/mathjax-kramdown
[haixing-hu]: https://gist.github.com/maurizzzio/ec311235997fab7b2993
[pages-gem]: https://github.com/github/pages-gem/issues/307
