Take a small model and a real loop, and make it tell you the truth about itself.

Print every tensor shape in the step, and write one line saying what each dimension means.

Verify one gradient by hand. Nudge a weight, measure how the loss changed, and compare
against what `backward()` reported. They should agree to several decimals, and if they do not,
you have found something worth understanding.

Break gradient accumulation on purpose. Use the average of the averages with micro-batches of
different lengths, and plot both curves together so you see the gap rather than take my word
for it.

Log the grad norm at every step, then find one step where it moved before the loss did.

Compute your own MFU, report it honestly, and say what you believe is costing you the distance
to 40%.

Take the number 0.1 and write out by hand what it looks like in fp32, bf16 and fp8 E4M3,
showing the bits. Then say which one you would train in, and why.

Print things and check things. Every serious training bug is silent, and the loss curve is not
going to be the one that tells you.

---

*Transcribed from Session 10 §17. Unlike Session 9, no separate assignment file was distributed —
the assignment exists only inside `s10-class-notes.html`, which is reference material and is not
committed. This file is that section, so the submission states what it was answering.*
