---
layout: post
title:  "How to reference equations properly in Microsoft Office (2013, 2016)"
date:   2017-03-27 09:59:59 +0100
categories: mypost
---
# Introduction

I haven't written anything for a long time. Sorry, my bad. However, I'm writing my PhD thesis these days and for some reasons I chose Microsoft Office to do that. Some might say, that LaTeX is much better, which is true in some sense. The main reason I chose Office is that I know it better, than the whole TeX system and I find much faster to type and format the material, even if it's not *that* pretty at the end. The figure and section references work quite well. I cannot even say very bad things about the referencing system. In Office 2013 I can choose the IEEE form (which was not present in the earlier versions like Office 2010), which is the common referencing format in my field. And using BibTex I can export my reference databases to Office 2007 format, which I can import in Office 2013/16. So things changed for the better over the years.

But there is always a *"but"*. And this but is about the equation referencing. The common solution for the equation numbering is to use brackets and a number in between, like the following (I use [MathJax][mathjax] here).

$$E = M \cdot c^2 \label{eq:matter-energy}$$

And when I later want to refer to that equation, I just use the reference and say, like, the number of the equation is \ref{eq:matter-energy}. I found a good guide about how to number your equations correctly and automatically ([this one][office2016-eq-num]), which explains it really well, but since it was a real nightmare until I succeeded, I try to give a longer, but more understandable version of it. Or rather let's just say it's my version of it. :) And I'd like to learn to use GitHub Pages and Jekyll. :) (Not to mention my English writing skills, as you can see reading the text above.) So in the end, after a long search and the trial-and-error method, I found a quite good solution, but I hope, that the future versions of Office will deal about this problem. I think, it wouldn't require much effort from the developers. In the first round, I would be happy, if these steps were work automatically, in one step.

# How to do it?

As far as I know, these steps work for both Office 2013 and 2016 versions (as of now).

* Insert an equation (either using Alt+E+C+B or clicking on Equation on the Insert tab) and type the equation you want.

![Insert equation]({{ site.github.url }}/images/2017-03-27/insert-equation-office-2016.png)

* There are two ways here. If you use Office 2016, you can easily make a choice like this:

![Equation numbering in Office 2016]({{ site.github.url }}/images/2017-03-27/numbering-office-2016.png).

After hitting Enter, you will get this nice layout, automatically:

![Office 2016 equation layout]({{ site.github.url }}/images/2017-03-27/office-2016-layout.png).

If you use Office 2013, you have to play with the tabulator settings, like this:

![Equation numbering in Office 2013]({{ site.github.url }}/images/2017-03-27/numbering-office-2013.png).

Note the two tabulator markers at the ruler. Only use them after you added the equation label, using the "Exclude label from caption" option, otherwise you can start placing the markers all over again.

* After you did this, if you reference to your equation using the normal reference settings, you will get the following result:

![Equation referencing bad]({{ site.github.url }}/images/2017-03-27/reference-numbering-bad.png).

And even if you delete everything else and leave just the equation number and the brackets there, if you update the fields in your document, it will change back to the form you see above. This does not look good at all! Let's do it better!

* Instead of using the method written in the previous point, it is better to use the bookmark option. After you completed the first two options from this list, you have to select the **(1)** part of the equation (the number may vary depending on the actual number of the equation, but that does not matter, since it is automatically generated). Then choose the Insert -> Bookmark option from the menu. Give it an identifier (like eq1 or eq_matter-energy), then click *Ok*. Now you can refer to the equation in the proper form using the References -> Cross-reference option from the menu. Here, choose the bookmark option from the dropdown-menu and the name of the equation bookmark. Then click *Ok*. You can see the result below.

![Equation referencing good]({{ site.github.url }}/images/2017-03-27/reference-numbering-good.png)

* There you go! Now if you update your fields in the document, it will look the same, don't worry. Here is an image about both the ugly and the pretty outcome. Since I have a Hungarian Office 2016, I did not want to insert the other widgets with buttons on them, since this post is in English. But if there is a request, I can add those figures as well. Just send me a PM. ;)

![Equation referencing both]({{ site.github.url }}/images/2017-03-27/reference-numbering-both.png)

[office2016-eq-num]: https://blogs.msdn.microsoft.com/murrays/2015/05/14/equation-numbering-in-office-2016/
[mathjax]: http://gastonsanchez.com/visually-enforced/opinion/2014/02/16/Mathjax-with-jekyll/
