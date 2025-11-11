cross_validation
================
Bohan Zhu
2025-11-11

``` r
library(tidyverse)
library(p8105.datasets)
library(modelr)

data("lidar")
```

Look at the data

``` r
lidar_df =
  lidar |> 
  mutate(id = row_number())

lidar_df |> 
  ggplot(aes(x = range, y = logratio)) +
  geom_point()
```

<img src="cross_validation_files/figure-gfm/unnamed-chunk-3-1.png" width="90%" />

## Create dataframes

``` r
train_df = 
  sample_frac(lidar_df, size = .8) |> 
  arrange(id)

test_df = anti_join(lidar_df, train_df, by = "id")
```

Look at these

``` r
ggplot(train_df, aes(x = range, y = logratio)) +
  geom_point()+
  geom_point(data = test_df, color = "red")
```

<img src="cross_validation_files/figure-gfm/unnamed-chunk-5-1.png" width="90%" />

Fit a few models to `train_df`

``` r
linear_mod = lm(logratio ~ range, data = train_df)

smooth_mod = mgcv::gam(logratio ~ s(range), data = train_df)

wiggly_mod = mgcv::gam(logratio ~ s(range, k = 30),  sp = 10e-6, data = train_df)
```

Look at this

``` r
train_df |> 
  add_predictions(wiggly_mod) |> 
  ggplot(aes(x = range, y =logratio)) +
  geom_point() +
  geom_line(aes(y = pred), color = "red")
```

<img src="cross_validation_files/figure-gfm/unnamed-chunk-7-1.png" width="90%" />

Try computing our RMSEs

``` r
rmse(linear_mod, test_df)
```

    ## [1] 0.131697

``` r
rmse(smooth_mod, test_df)
```

    ## [1] 0.08902606

``` r
rmse(wiggly_mod, test_df)
```

    ## [1] 0.100557

## ITERATE!!

``` r
cv_df = 
  crossv_mc(lidar_df, n = 100) |> 
  mutate(
    train = map(train, as_tibble),
    test = map(test, as_tibble)
  )
```

Did this work?

``` r
cv_df |> 
  pull(train) |> nth(3)
```

    ## # A tibble: 176 × 3
    ##    range logratio    id
    ##    <dbl>    <dbl> <int>
    ##  1   393  -0.0419     3
    ##  2   396  -0.0599     5
    ##  3   397  -0.0284     6
    ##  4   400  -0.0399     8
    ##  5   402  -0.0294     9
    ##  6   403  -0.0395    10
    ##  7   405  -0.0476    11
    ##  8   406  -0.0604    12
    ##  9   408  -0.0312    13
    ## 10   409  -0.0382    14
    ## # ℹ 166 more rows

Let’s fit models over and over

``` r
lidar_lm = function(df){
  lm(logratio ~range, data = df)
}
```

``` r
cv_df =
  cv_df |> 
  mutate(
    linear_fits = map(train, \(df) lm(logratio ~ range, data = df)),
    smooth_fits = map(train, \(df) mgcv::gam(logratio ~ s(range), data = df)),
    wiggly_fits = map(train, \(df)mgcv::gam(logratio ~ s(range, k =50), 
                                            sp = 10e-8, data = df))
  ) |> 
  mutate(
    rmse_line = map2_dbl(linear_fits, test, rmse),
    rmse_smooth = map2_dbl(smooth_fits, test, rmse),
    rmse_wiggly = map2_dbl(wiggly_fits, test, rmse)
  )
```

Let’s try to look at this better

``` r
cv_df |> 
  select(starts_with("rmse")) |> 
  pivot_longer(
    everything(),
    names_to = "model",
    values_to = "rmse",
    names_prefix = "rmse_"
  ) |> 
  ggplot(aes(x = model, y = rmse,fill = model)) +
  geom_violin()
```

<img src="cross_validation_files/figure-gfm/unnamed-chunk-13-1.png" width="90%" />
