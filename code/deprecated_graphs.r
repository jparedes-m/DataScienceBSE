# deprecated graphs from jorge
## [3.1] Univariate analysis ----
data_long <- data %>% select(sex, age, credit_amount, duration) %>%
  pivot_longer(cols = c(age, credit_amount, duration), names_to = "variable", values_to = "value") %>% 
  mutate(variable = case_when(
    variable == "age" ~ "Age (in years)",
    variable == "credit_amount" ~ "Credit Amount (in DM)",
    variable == "duration" ~ "Duration (in months)"))

ggplot(data_long, aes(x = value, fill = sex)) +
  geom_histogram(aes(y = after_stat(density)), position = "identity", bins = 30, alpha = 0.5) +
  geom_density(aes(color = sex), linewidth = 1, fill = NA) +
  labs(y = "Density", x = " ", title = "Distribution by Sex", fill = "Sex:", color = "Sex:") +
  theme_light() +
  scale_x_continuous(n.breaks = 17) +
  scale_y_continuous(n.breaks = 10) +
  facet_wrap(~ variable, scales = "free") +
  scale_fill_manual(values = c("male" = "cornflowerblue", "female" = "firebrick")) +
  scale_color_manual(values = c("male" = "cornflowerblue", "female" = "firebrick")) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 15),
        axis.title.y = element_text(size = 12),
        axis.title.x = element_text(size = 12), 
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),
        strip.text.x = element_text(face = "bold", color = "black", size = 12),
        legend.position = "bottom")

#ggsave("Aplicaciones/Overleaf/Foundations of Data Science - BSE Group/assets/figures/3_1_graph.png", width = 9, height = 10, bg = "white")
dev.off()
## [3.2] Credit amount ----
data %>% mutate(p_status = ifelse(p_status == "single"| p_status == "div/sep", "Single", "Married")) %>% 
ggplot(aes(x = credit_amount)) +
    geom_histogram(aes(y = after_stat(density)), bins = 30, fill = "dodgerblue3", alpha = 0.7) +
    geom_density(color = "blue", linewidth = 1) +
    labs(y = "Density", x = "Credit Amount in DM", title = "Credit Amount Distribution by Status") +
    theme_light() +
    scale_x_continuous(n.breaks = 20) +
    scale_y_continuous(n.breaks = 10) +
    theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 15),
          strip.text.x = element_text(face = "bold", color = "black", size = 12),
          axis.title.y = element_text(size = 12),
          axis.title.x = element_text(size = 12),
          axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
    facet_wrap(~ p_status, scales = "free", nrow = 2)

#ggsave("Aplicaciones/Overleaf/Foundations of Data Science - BSE Group/assets/figures/3_2_1_graph.png", width = 9, height = 10, bg = "white")

data %>%
  mutate(p_status = ifelse(p_status == "single" | p_status == "div/sep", "Single", "Married")) %>%
  ggplot(aes(x = credit_amount, fill = p_status)) +
  geom_histogram(aes(y = after_stat(density)), bins = 30, alpha = 0.7, position = "identity") +
  geom_density(aes(color = p_status), linewidth = 1, fill = NA) +
  labs(
    y = "Density",
    x = "Credit Amount in DM",
    title = "Credit Amount Distribution by Marital Status",
    fill = "Marital Status:",
    color = "Marital Status:"
  ) +
  theme_light() +
  scale_x_continuous(n.breaks = 20) +
  scale_y_continuous(n.breaks = 10) +
  scale_fill_manual(values = c("Single" = "cornflowerblue", "Married" = "firebrick")) +
  scale_color_manual(values = c("Single" = "cornflowerblue", "Married" = "firebrick")) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 15),
    axis.title.y = element_text(size = 12),
    axis.title.x = element_text(size = 12),
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
    strip.text.x = element_text(face = "bold", color = "black", size = 12),
    legend.position = "bottom"
  )

#ggsave("Aplicaciones/Overleaf/Foundations of Data Science - BSE Group/assets/figures/3_2_2_graph.png", width = 9, height = 10, bg = "white")

ggplot(data, aes(x = credit_amount)) +
    geom_histogram(aes(y = after_stat(density)), bins = 30, fill = "dodgerblue3", alpha = 0.7) +
    geom_density(color = "blue", linewidth = 1) +
    labs(y = "Density", x = "Credit Amount in DM", title = "Credit Amount Distribution by product") +
    theme_light() +
    scale_x_continuous(n.breaks = 20) +
    scale_y_continuous(n.breaks = 10) +
    theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 15),
          strip.text.x = element_text(face = "bold", color = "black", size = 12),
          axis.title.y = element_text(size = 12),
          axis.title.x = element_text(size = 12),
          axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
    facet_wrap(~ purpose, scales = "free", nrow = 2)

#ggsave("Aplicaciones/Overleaf/Foundations of Data Science - BSE Group/assets/figures/3_2_3_graph.png", width = 15, height = 10, bg = "white")
dev.off()
data %>% mutate(class = ifelse(class == 0, "Good", "Bad")) %>%
ggplot(aes(x = credit_amount, fill = sex)) +
    geom_histogram(aes(y = after_stat(density)), bins = 30, alpha = 0.7, position = "identity") +
    geom_density(aes(color = sex), linewidth = 1, fill = NA) +
    labs(
        y = "Density",
        x = "Credit Amount in DM",
        title = "Credit Amount Distribution by Class and Sex",
        fill = "Sex",
        color = "Sex"
    ) +
    theme_light() +
    scale_x_continuous(n.breaks = 20) +
    scale_y_continuous(n.breaks = 10) +
    theme(
        plot.title = element_text(hjust = 0.5, face = "bold", size = 15),
        strip.text.x = element_text(face = "bold", color = "black", size = 12),
        axis.title.y = element_text(size = 12),
        axis.title.x = element_text(size = 12),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
        legend.position = "bottom") +
    facet_wrap(~ class, scales = "free", nrow = 2) +
    scale_fill_manual(values = c("male" = "cornflowerblue", "female" = "firebrick")) +
    scale_color_manual(values = c("male" = "cornflowerblue", "female" = "firebrick"))
dev.off()
#ggsave("Aplicaciones/Overleaf/Foundations of Data Science - BSE Group/assets/figures/3_2_4_graph.png", width = 9, height = 10, bg = "white")

## [3.3] Duration ----
data %>% mutate(p_status = ifelse(p_status == "single"| p_status == "div/sep", "Single", "Married")) %>% 
ggplot(aes(x = duration)) +
    geom_histogram(aes(y = after_stat(density)), bins = 30, fill = "dodgerblue3", alpha = 0.7) +
    geom_density(color = "blue", linewidth = 1) +
    labs(y = "Density", x = "Duration (in months)", title = "Credit Duration Distribution by Status") +
    theme_light() +
    scale_x_continuous(n.breaks = 20) +
    scale_y_continuous(n.breaks = 10) +
    theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 15),
          strip.text.x = element_text(face = "bold", color = "black", size = 12),
          axis.title.y = element_text(size = 12),
          axis.title.x = element_text(size = 12),
          axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
    facet_wrap(~ p_status, scales = "free", nrow = 2)

#ggsave("Aplicaciones/Overleaf/Foundations of Data Science - BSE Group/assets/figures/3_3_1_graph.png", width = 9, height = 10, bg = "white")
dev.off()
data %>%
  mutate(p_status = ifelse(p_status == "single" | p_status == "div/sep", "Single", "Married")) %>%
  ggplot(aes(x = duration, fill = p_status)) +
  geom_histogram(aes(y = after_stat(density)), bins = 30, alpha = 0.7, position = "identity") +
  geom_density(aes(color = p_status), linewidth = 1, fill = NA) +
  labs(
    y = "Density",
    x = "Duration (in months)",
    title = "Credit Duration Distribution by Marital Status",
    fill = "Marital Status:",
    color = "Marital Status:"
  ) +
  theme_light() +
  scale_x_continuous(n.breaks = 20) +
  scale_y_continuous(n.breaks = 10) +
  scale_fill_manual(values = c("Single" = "cornflowerblue", "Married" = "firebrick")) +
  scale_color_manual(values = c("Single" = "cornflowerblue", "Married" = "firebrick")) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 15),
    axis.title.y = element_text(size = 12),
    axis.title.x = element_text(size = 12),
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
    strip.text.x = element_text(face = "bold", color = "black", size = 12),
    legend.position = "bottom"
  )
dev.off()
#ggsave("Aplicaciones/Overleaf/Foundations of Data Science - BSE Group/assets/figures/3_3_2_graph.png", width = 9, height = 10, bg = "white")

ggplot(data, aes(x = duration)) +
    geom_histogram(aes(y = after_stat(density)), bins = 30, fill = "dodgerblue3", alpha = 0.7) +
    geom_density(color = "blue", linewidth = 1) +
    labs(y = "Density", x = "Duration (in months)", title = "Credit Duration Distribution by product") +
    theme_light() +
    scale_x_continuous(n.breaks = 20) +
    scale_y_continuous(n.breaks = 10) +
    theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 15),
          strip.text.x = element_text(face = "bold", color = "black", size = 12),
          axis.title.y = element_text(size = 12),
          axis.title.x = element_text(size = 12),
          axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
    facet_wrap(~ purpose, scales = "free", nrow = 2)

#ggsave("Aplicaciones/Overleaf/Foundations of Data Science - BSE Group/assets/figures/3_3_3_graph.png", width = 15, height = 10, bg = "white")
dev.off()
data %>% mutate(class = ifelse(class == 0, "Good", "Bad")) %>%
ggplot(aes(x = duration, fill = sex)) +
    geom_histogram(aes(y = after_stat(density)), bins = 30, alpha = 0.7, position = "identity") +
    geom_density(aes(color = sex), linewidth = 1, fill = NA) +
    labs(
        y = "Density",
        x = "Duration (in months)",
        title = "Credit Duration Distribution by Class and Sex",
        fill = "Sex",
        color = "Sex"
    ) +
    theme_light() +
    scale_x_continuous(n.breaks = 20) +
    scale_y_continuous(n.breaks = 10) +
    theme(
        plot.title = element_text(hjust = 0.5, face = "bold", size = 15),
        strip.text.x = element_text(face = "bold", color = "black", size = 12),
        axis.title.y = element_text(size = 12),
        axis.title.x = element_text(size = 12),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
        legend.position = "bottom") +
    facet_wrap(~ class, scales = "free", nrow = 2) +
    scale_fill_manual(values = c("male" = "cornflowerblue", "female" = "firebrick")) +
    scale_color_manual(values = c("male" = "cornflowerblue", "female" = "firebrick"))
dev.off()
#ggsave("Aplicaciones/Overleaf/Foundations of Data Science - BSE Group/assets/figures/3_3_4_graph.png", width = 9, height = 10, bg = "white")
