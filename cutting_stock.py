import tkinter as tk
from tkinter import messagebox
import pulp
import matplotlib.pyplot as plt
import math
import matplotlib
matplotlib.use("TkAgg")

def generate_initial_patterns(lengths, stock_length):
    patterns = []
    for p in lengths:
        count = int(stock_length // p)
        if count > 0:
            pattern = {x: 0 for x in lengths}
            pattern[p] = count
            patterns.append(pattern)
    return patterns

def solve_master_lp(patterns, demand):
    prob = pulp.LpProblem("CuttingStockMaster", pulp.LpMinimize)
    x_vars = [pulp.LpVariable(f"x_{i}", lowBound=0, cat="Continuous") for i in range(len(patterns))]

    prob += pulp.lpSum(x_vars)

    for p in demand:
        prob += (
            pulp.lpSum(x_vars[i] * patterns[i][p] for i in range(len(patterns))) >= demand[p],
            f"Demand_{p}"
        )

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    return prob, x_vars

def knapsack_dp(dual_values, lengths, stock_length):
    n = len(lengths)
    max_len = int(stock_length * 1000)
    scaled_lengths = [int(l * 1000) for l in lengths]
    dp = [-float('inf')] * (max_len + 1)
    dp[0] = 0
    choice = [-1] * (max_len + 1)

    for i in range(n):
        length_i = scaled_lengths[i]
        value_i = dual_values.get(lengths[i], 0)
        for w in range(length_i, max_len + 1):
            if dp[w - length_i] + value_i > dp[w]:
                dp[w] = dp[w - length_i] + value_i
                choice[w] = i

    max_value = max(dp)
    w = dp.index(max_value)
    counts = {p: 0 for p in lengths}
    while w > 0 and choice[w] != -1:
        i = choice[w]
        counts[lengths[i]] += 1
        w -= scaled_lengths[i]
    reduced_cost = 1 - max_value
    return counts, reduced_cost

def column_generation(lengths, demand, stock_length, max_iter=100):
    patterns = generate_initial_patterns(lengths, stock_length)
    for iteration in range(max_iter):
        master_prob, x_vars = solve_master_lp(patterns, demand)
        if pulp.LpStatus[master_prob.status] != "Optimal":
            messagebox.showerror("Hata", "Master problem optimal çözüm bulamadı.")
            return None, None

        x_values = [x_vars[i].varValue if x_vars[i].varValue is not None else 0 for i in range(len(patterns))]
        duals = {}
        for name, constraint in master_prob.constraints.items():
            p_str = name.replace("Demand_", "")
            try:
                p = float(p_str)
            except:
                p = p_str
            dual_val = constraint.pi
            if dual_val is None:
                dual_val = 0
            duals[p] = dual_val

        new_pattern, reduced_cost = knapsack_dp(duals, lengths, stock_length)
        if reduced_cost >= -1e-6:
            return patterns, x_values
        if all(new_pattern.get(p, 0) == patterns[-1].get(p, 0) for p in lengths):
            return patterns, x_values
        patterns.append(new_pattern)
    messagebox.showwarning("Uyarı", "Maksimum iterasyona ulaşıldı, tam optimal olmayabilir.")
    return patterns, x_values

def round_and_repair(patterns, x_values, lengths, demand, stock_length):
    import numpy as np
    x_rounded = [math.floor(x) for x in x_values]
    total_covered = {p: 0 for p in lengths}
    for i, val in enumerate(x_rounded):
        for p in lengths:
            total_covered[p] += val * patterns[i].get(p, 0)
    residual = {p: max(0, demand[p] - total_covered[p]) for p in lengths}
    if all(v == 0 for v in residual.values()):
        fire_list = []
        for i, count in enumerate(x_rounded):
            used_length = sum([patterns[i].get(p, 0) * p for p in lengths])
            fire = stock_length - used_length
            fire_list.append(fire)
        return x_rounded, patterns, fire_list

    new_patterns = []
    new_solution = []
    res_keys = list(residual.keys())
    res_values = [residual[k] for k in res_keys]
    max_len_mm = int(stock_length * 1000)
    lengths_mm = [int(l * 1000) for l in res_keys]
    while any(v > 0 for v in res_values):
        capacity = max_len_mm
        dp = [-1] * (capacity + 1)
        dp[0] = 0
        choice = [-1] * (capacity + 1)
        max_counts = [int(v) for v in res_values]
        for i, length_i in enumerate(lengths_mm):
            count = max_counts[i]
            k = 1
            while count > 0:
                use = min(k, count)
                count -= use
                length_use = length_i * use
                value_use = use
                for w in range(capacity, length_use -1, -1):
                    if dp[w - length_use] != -1 and dp[w - length_use] + value_use > dp[w]:
                        dp[w] = dp[w - length_use] + value_use
                        choice[w] = (i, use)
                k *= 2
        max_value = max(dp)
        if max_value <= 0:
            messagebox.showerror("Hata", "Kalan talepler tam olarak karşılanamadı!")
            return None, None, None
        w = dp.index(max_value)
        pattern_counts = {k: 0 for k in res_keys}
        cur_w = w
        while cur_w > 0 and choice[cur_w] != -1:
            i, use = choice[cur_w]
            pattern_counts[res_keys[i]] += use
            cur_w -= lengths_mm[i] * use
        for i, k in enumerate(res_keys):
            res_values[i] = max(0, res_values[i] - pattern_counts[k])
        new_patterns.append(pattern_counts)
        new_solution.append(1)
    final_solution = x_rounded + new_solution
    final_patterns = patterns + new_patterns
    fire_list = []
    for i, count in enumerate(final_solution):
        used_length = sum([final_patterns[i].get(p, 0) * p for p in lengths])
        fire = stock_length - used_length
        fire_list.append(fire)
    return final_solution, final_patterns, fire_list

def draw_result(patterns, solution, stock_length, fire_list):
    birim = "m"
    aktif = [(i, adet) for i, adet in enumerate(solution) if adet > 0]
    if not aktif:
        messagebox.showinfo("Bilgi", "Kullanılan desen bulunamadı.")
        return
    fig, ax = plt.subplots(figsize=(10, max(6, len(aktif) * 0.7)))
    y_pos = []
    labels = []
    renkler = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']
    for idx, (i, adet) in enumerate(aktif):
        y = idx
        y_pos.append(y)
        labels.append(f"Desen {i + 1} - {int(adet)} adet")
        baslangic = 0
        toplam_uzunluk = 0
        for ridx, (parca, sayi) in enumerate(patterns[i].items()):
            if sayi > 0:
                uzunluk = parca * sayi
                ax.barh(y, uzunluk, left=baslangic, height=0.5,
                        color=renkler[ridx % len(renkler)], edgecolor='black')
                ax.text(baslangic + uzunluk / 2, y, f"{sayi}x{parca}{birim}",
                        va='center', ha='center', color='white', fontsize=9)
                baslangic += uzunluk
                toplam_uzunluk += uzunluk
        fire = fire_list[i]
        if fire > 0:
            ax.barh(y, fire, left=baslangic, height=0.5, color='lightgrey', edgecolor='black')
            ax.text(baslangic + fire / 2, y, f"Fire: {fire:.3f}{birim}",
                    va='center', ha='center', color='black', fontsize=9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel(f"Uzunluk ({birim})")
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show(block=True)

def hesapla():
    try:
        stock_length = float(stock_length_entry.get())
        piece_lengths = [float(e.get()) for e in piece_length_entries]
        demands = [int(e.get()) for e in demand_entries]
    except Exception as e:
        messagebox.showerror("Hata", f"Girdi hatası: {e}")
        return
    if len(piece_lengths) != len(demands):
        messagebox.showerror("Hata", "Parça boyları ve talepler aynı sayıda olmalıdır!")
        return
    demand_dict = dict(zip(piece_lengths, demands))
    result = column_generation(piece_lengths, demand_dict, stock_length)
    if result == (None, None):
        return
    patterns, x_values = result
    solution, patterns_updated, fire_list = round_and_repair(patterns, x_values, piece_lengths, demand_dict, stock_length)
    if solution is None or patterns_updated is None:
        return
    output_text.config(state=tk.NORMAL)
    output_text.delete("1.0", tk.END)
    toplam_stok = sum(solution)
    output_text.insert(tk.END, f"Toplam kullanılan ana parça sayısı: {toplam_stok}\n\n")
    for idx, count in enumerate(solution):
        if count > 0:
            pattern_desc = ", ".join(f"{int(v)}x{float(k)}" for k, v in patterns_updated[idx].items() if v > 0)
            output_text.insert(tk.END, f"Desen {idx+1} (adet {count}): {pattern_desc}\n")
    output_text.config(state=tk.DISABLED)
    for widget in plot_frame.winfo_children():
        widget.destroy()
    draw_result(patterns_updated, solution, stock_length, fire_list)


root = tk.Tk()
root.title("Cutting Stock Problem - Column Generation + Round & Repair")
root.geometry("700x600")
root.minsize(700, 600)
canvas = tk.Canvas(root, borderwidth=0, highlightthickness=0)
scroll_frame = tk.Frame(canvas)
vsb = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
canvas.configure(yscrollcommand=vsb.set)
vsb.pack(side="right", fill="y")
canvas.pack(side="left", fill="both", expand=True)
canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
tk.Label(scroll_frame, text="Ana Parça Uzunluğu (metre):", font=("Arial", 12)).pack(pady=5)
stock_length_entry = tk.Entry(scroll_frame, font=("Arial", 12))
stock_length_entry.pack(pady=2)
stock_length_entry.insert(0, "30")
frame_inputs = tk.Frame(scroll_frame)
frame_inputs.pack(pady=10, fill="x")
tk.Label(frame_inputs, text="Parça Boyu", font=("Arial", 12)).grid(row=0, column=0, padx=5, sticky="nsew")
tk.Label(frame_inputs, text="Talep", font=("Arial", 12)).grid(row=0, column=1, padx=5, sticky="nsew")
piece_length_entries = []
demand_entries = []
delete_buttons = []
def add_row():
    row = len(piece_length_entries) + 1
    ple = tk.Entry(frame_inputs, width=10, font=("Arial", 11))
    ple.grid(row=row, column=0, padx=5, pady=2, sticky="nsew")
    ple.bind("<FocusIn>", lambda e: e.widget.select_range(0, tk.END))
    de = tk.Entry(frame_inputs, width=10, font=("Arial", 11))
    de.grid(row=row, column=1, padx=5, pady=2, sticky="nsew")
    de.bind("<FocusIn>", lambda e: e.widget.select_range(0, tk.END))
    piece_length_entries.append(ple)
    demand_entries.append(de)
    def delete_row():
        idx = piece_length_entries.index(ple)
        ple.grid_forget()
        de.grid_forget()
        del_btn.grid_forget()
        piece_length_entries.pop(idx)
        demand_entries.pop(idx)
        delete_buttons.pop(idx)
        for i, (pl_e, d_e) in enumerate(zip(piece_length_entries, demand_entries), start=1):
            pl_e.grid(row=i, column=0, sticky="nsew", padx=5, pady=2)
            d_e.grid(row=i, column=1, sticky="nsew", padx=5, pady=2)
        for i, btn in enumerate(delete_buttons, start=1):
            btn.grid(row=i, column=2, sticky="nsew", padx=5, pady=2, ipadx=10, ipady=5)
    del_btn = tk.Button(frame_inputs, text="Sil", font=("Arial", 11, "bold"), fg="red",
                        command=delete_row, cursor="hand2",
                        relief="raised", borderwidth=3)
    del_btn.grid(row=row, column=2, padx=5, pady=2, sticky="nsew", ipadx=10, ipady=5)
    delete_buttons.append(del_btn)
for _ in range(3):
    add_row()
btn_frame = tk.Frame(scroll_frame)
btn_frame.pack(pady=10)
btn_add_row = tk.Button(btn_frame, text="Yeni Satır Ekle", font=("Arial", 14, "bold"), command=add_row, cursor="hand2",
                        relief="raised", borderwidth=4, height=2, width=15)
btn_add_row.grid(row=0, column=0, padx=10)
btn_calculate = tk.Button(btn_frame, text="Hesapla ve Göster", font=("Arial", 14, "bold"), command=hesapla, cursor="hand2",
                          relief="raised", borderwidth=4, height=2, width=18)
btn_calculate.grid(row=0, column=1, padx=10)
output_text = tk.Text(scroll_frame, height=15, width=70, font=("Arial", 11), state=tk.DISABLED, wrap="word")
output_text.pack(pady=5)
plot_frame = tk.Frame(scroll_frame)
plot_frame.pack(fill=tk.BOTH, expand=True)
root.mainloop()
 
