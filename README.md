# opt_for_pinns

## `sksgd_line_search_lsq.ipynb`

To enable line-search feature in SketchySGD, pass in `line_search_fn = backtracking` for back-tracking line search (using Armijo rule) or `line_search_fn = strong_wolfe` for Strong Wolfe (implementation largely follows PyTorch's L-BFGS). Setting argument `line_search_fn` to `None` turns off line-search feature altogether. 

The `use_interpolation` argument in `self._backtracking` controls if quadratic interpolation or basic constant scaling factor is to be used. 

The `verbose` argument of the SketchySGD allows Hessian approximation to be printed out. 
