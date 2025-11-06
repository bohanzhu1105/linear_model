linear_model
================
Bohan Zhu
2025-11-06

``` r
library(tidyverse)
```

    ## ── Attaching core tidyverse packages ──────────────────────── tidyverse 2.0.0 ──
    ## ✔ dplyr     1.1.4     ✔ readr     2.1.5
    ## ✔ forcats   1.0.0     ✔ stringr   1.5.1
    ## ✔ ggplot2   4.0.0     ✔ tibble    3.3.0
    ## ✔ lubridate 1.9.4     ✔ tidyr     1.3.1
    ## ✔ purrr     1.1.0     
    ## ── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
    ## ✖ dplyr::filter() masks stats::filter()
    ## ✖ dplyr::lag()    masks stats::lag()
    ## ℹ Use the conflicted package (<http://conflicted.r-lib.org/>) to force all conflicts to become errors

``` r
library(p8105.datasets)
```

``` r
data(nyc_airbnb)

nyc_airbnb =
  nyc_airbnb |> 
  mutate(
    stars = review_scores_location / 2
  ) |> 
  rename(
    borough = neighbourhood_group
  ) |> 
  filter(borough != "Staten Island") |> 
  select(price, stars, borough, room_type, neighbourhood)
```

Do regression model

``` r
fit = lm(price ~ stars + borough, data = nyc_airbnb)
```

Do some additional cleaning then refit.

``` r
nyc_airbnb = 
  nyc_airbnb |> 
  mutate(
    borough = fct_infreq(borough),
    room_typ = fct_infreq(room_type) # fct_infreq() puting categorics in frequency order to relevel the fector level
  )

fit = lm(price ~ stars + borough, data = nyc_airbnb)
```

Look at `lm` stuff

``` r
summary(fit)
names(summary(fit))
summary(fit)[["coefficients"]]
summary(fit)[["df"]]

fitted.values(fit)
```

Looking at cleaner `lm` stuff

``` r
fit |> 
  broom::tidy() |> 
  mutate(
    term = str_replace(term, "borough", "Borough:")
  ) |> 
  select(term,estimate,p.value) |> 
  knitr::kable(digits = 3)
```

| term             | estimate | p.value |
|:-----------------|---------:|--------:|
| (Intercept)      |   19.839 |   0.104 |
| stars            |   31.990 |   0.000 |
| Borough:Brooklyn |  -49.754 |   0.000 |
| Borough:Queens   |  -77.048 |   0.000 |
| Borough:Bronx    |  -90.254 |   0.000 |

``` r
fit |> 
  broom::glance()
```

    ## # A tibble: 1 × 12
    ##   r.squared adj.r.squared sigma statistic   p.value    df   logLik    AIC    BIC
    ##       <dbl>         <dbl> <dbl>     <dbl>     <dbl> <dbl>    <dbl>  <dbl>  <dbl>
    ## 1    0.0342        0.0341  182.      271. 6.73e-229     4 -202113. 4.04e5 4.04e5
    ## # ℹ 3 more variables: deviance <dbl>, df.residual <int>, nobs <int>
