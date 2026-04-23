library(vctrs)
library(dplyr)
library(haven)

read_nhanes_xpt_public <- function(stem, letter, public_year){
  url <- sprintf(
    "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/%s/DataFiles/%s_%s.XPT",
    public_year, stem, letter
  )
  tmp <- tempfile(fileext = ".XPT")
  download.file(url, tmp, mode = "wb", quiet = TRUE)

  hdr <- rawToChar(readBin(tmp, "raw", 80))
  if (!grepl("HEADER RECORD", hdr, fixed = TRUE)) {
    stop("Downloaded file is not an XPT. URL may be wrong:\n", url)
  }
  read_xpt(tmp)
}

summarize_cycle <- function(label, letter, public_year){
  DEMO <- read_nhanes_xpt_public("DEMO", letter, public_year)
  DIQ  <- read_nhanes_xpt_public("DIQ",  letter, public_year)
  GHB  <- read_nhanes_xpt_public("GHB",  letter, public_year)
  GLU  <- read_nhanes_xpt_public("GLU",  letter, public_year)

  dat <- DEMO %>%
    left_join(DIQ, by="SEQN") %>%
    left_join(GHB, by="SEQN") %>%
    left_join(GLU, by="SEQN")

  has_WTSAF2YR <- "WTSAF2YR" %in% names(dat)
  has_WTMEC2YR <- "WTMEC2YR" %in% names(dat)

  dat <- dat %>%
    mutate(
      Y = case_when(DIQ010 == 1 ~ 1L,
                    DIQ010 == 2 ~ 0L,
                    TRUE ~ NA_integer_),
      D_a1c = case_when(!is.na(LBXGH) & LBXGH >= 6.5 ~ 1L,
                        !is.na(LBXGH)               ~ 0L,
                        TRUE ~ NA_integer_),
      fasting_ok = if (has_WTSAF2YR) (!is.na(WTSAF2YR) & WTSAF2YR > 0) else NA
    )

  examined_n <- if (has_WTMEC2YR) sum(dat$WTMEC2YR > 0, na.rm=TRUE) else NA_integer_

  und <- dat %>% filter(D_a1c==1, !is.na(Y)) %>% summarise(rate=mean(Y==0), n=n())

  tibble(
    cycle = label,
    interview_n = nrow(DEMO),
    examined_n  = examined_n,
    a1c_n       = sum(!is.na(dat$LBXGH)),
    fasting_glu_n = sum(!is.na(dat$LBXGLU)),
    Y_n         = sum(!is.na(dat$Y)),
    undiagnosed_rate_a1c = und$rate,
    undiagnosed_n_a1c    = und$n
  )
}

res <- bind_rows(
  summarize_cycle("2017-2018 (J)", "J", "2017"),
  summarize_cycle("2021-2023 (L)", "L", "2021")
)

print(res)

read_many <- function(letter, public_year){
  stems <- c("DEMO","DIQ","GHB","BMX","HIQ","HUQ","INQ")
  dfs <- lapply(stems, \(s) read_nhanes_xpt_public(s, letter, public_year))
  names(dfs) <- stems
  Reduce(\(x,y) left_join(x,y, by="SEQN"), dfs)
}

datJ <- read_many("J","2017")

datJ2 <- datJ %>%
  mutate(
    Y = case_when(DIQ010==1 ~ 1L, DIQ010==2 ~ 0L, TRUE ~ NA_integer_),
    D_a1c = case_when(!is.na(LBXGH) & LBXGH >= 6.5 ~ 1L,
                      !is.na(LBXGH) ~ 0L, TRUE ~ NA_integer_),
    insured = case_when(HIQ011==1 ~ 1L, HIQ011==2 ~ 0L, TRUE ~ NA_integer_),
    usualcare = case_when(HUQ030==1 ~ 1L, HUQ030==2 ~ 0L, TRUE ~ NA_integer_),
    female = case_when(RIAGENDR==2 ~ 1L, RIAGENDR==1 ~ 0L, TRUE ~ NA_integer_)
  )

fit_diag <- glm(Y ~ insured + usualcare + RIDAGEYR + BMXBMI + female,
                data = datJ2 %>% filter(D_a1c==1, !is.na(Y)),
                family = binomial())

summary(fit_diag)

df0 <- datJ2 %>%
  transmute(
    Y = Y,
    insured = insured,
    usualcare = usualcare,
    age = as.numeric(RIDAGEYR),
    bmi = as.numeric(BMXBMI),
    female = female
  ) %>%
  filter(!is.na(Y), !is.na(insured), !is.na(usualcare),
         !is.na(age), !is.na(bmi), !is.na(female)) %>%
  mutate(
    age_z = as.numeric(scale(age)),
    bmi_z = as.numeric(scale(bmi))
  )

fit_zi_xeqz <- function(df){
  X <- model.matrix(~ insured + usualcare + age_z + bmi_z + female, data=df)
  y <- df$Y
  d <- ncol(X)
  eps <- 1e-12

  nll <- function(par){
    beta  <- par[1:d]
    gamma <- par[(d+1):(2*d)]
    p1 <- plogis(X %*% beta) * plogis(X %*% gamma)
    -sum(y*log(p1+eps) + (1-y)*log(1-p1+eps))
  }

  run_once <- function(start){
    optim(start, nll, method="BFGS", control=list(maxit=3000))
  }

  set.seed(1)
  start1 <- rnorm(2*d, 0, 0.1)
  out1 <- run_once(start1)

  start2 <- c(out1$par[(d+1):(2*d)], out1$par[1:d])
  out2 <- run_once(start2)

  list(X=X, out1=out1, out2=out2)
}

res <- fit_zi_xeqz(df0)

c(nll1=res$out1$value, nll2=res$out2$value)

X <- res$X; d <- ncol(X)
b1 <- res$out1$par[1:d];      g1 <- res$out1$par[(d+1):(2*d)]
b2 <- res$out2$par[1:d];      g2 <- res$out2$par[(d+1):(2*d)]

c(max_abs_beta1_minus_gamma2 = max(abs(b1 - g2)),
  max_abs_gamma1_minus_beta2 = max(abs(g1 - b2)))

fit_lr <- glm(Y ~ insured + usualcare + age_z + bmi_z + female,
              data=df0, family=binomial())
b_lr <- coef(fit_lr)

X <- res$X; d <- ncol(X)
b1 <- res$out1$par[1:d]; b2 <- res$out2$par[1:d]

dist1 <- sum((b1 - b_lr)^2)
dist2 <- sum((b2 - b_lr)^2)
c(dist1=dist1, dist2=dist2)

X <- res$X; d <- ncol(X)
b1 <- res$out1$par[1:d]; g1 <- res$out1$par[(d+1):(2*d)]
b2 <- res$out2$par[1:d]; g2 <- res$out2$par[(d+1):(2*d)]

tab <- data.frame(
  term = colnames(X),
  beta_1 = b1, gamma_1 = g1,
  beta_2 = b2, gamma_2 = g2,
  check_beta1_minus_gamma2 = b1 - g2,
  check_gamma1_minus_beta2 = g1 - b2
)
tab

