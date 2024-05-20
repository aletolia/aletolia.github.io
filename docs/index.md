---
template: home.html
hide:
  - navigation
  - toc
  - footer
---

<div id="rcorners2" >
  <div id="rcorners1">
    <!-- <i class="fa fa-calendar" style="font-size:100"></i> -->
    <body>
      <font color="#4351AF">
        <p class="p1"></p>
<script defer>
    //格式：2020年04月12日 10:20:00 星期二
    function format(newDate) {
        var day = newDate.getDay();
        var y = newDate.getFullYear();
        var m =
            newDate.getMonth() + 1 < 10
                ? "0" + (newDate.getMonth() + 1)
                : newDate.getMonth() + 1;
        var d =
            newDate.getDate() < 10 ? "0" + newDate.getDate() : newDate.getDate();
        var h =
            newDate.getHours() < 10 ? "0" + newDate.getHours() : newDate.getHours();
        var min =
            newDate.getMinutes() < 10
                ? "0" + newDate.getMinutes()
                : newDate.getMinutes();
        var s =
            newDate.getSeconds() < 10
                ? "0" + newDate.getSeconds()
                : newDate.getSeconds();
        var dict = {
            1: "一",
            2: "二",
            3: "三",
            4: "四",
            5: "五",
            6: "六",
            0: "天",
        };
        //var week=["日","一","二","三","四","五","六"]
        return (
            y +
            "年" +
            m +
            "月" +
            d +
            "日" +
            " " +
            h +
            ":" +
            min +
            ":" +
            s +
            " 星期" +
            dict[day]
        );
    }
    var timerId = setInterval(function () {
        var newDate = new Date();
        var p1 = document.querySelector(".p1");
        if (p1) {
            p1.textContent = format(newDate);
        }
    }, 1000);
</script>
      </font>
    </body>

此网站主要做储存资料用，包括单细胞，计算机视觉以及多模态方面的文献，也包含以往的教材翻译

<!-- <div class="grid cards" markdown>

-   :octicons-bookmark-16:{ .lg .middle } __scRNA-seq__

    ---

    - [:octicons-arrow-right-24:scGPT](scGPT toward building a foundation model for single-cell multi-omics using generative AI.md)
    - [:octicons-arrow-right-24:泛癌B细胞建模](A blueprint for tumor-infiltrating B cells across human cancers.md)
    - [:octicons-arrow-right-24:泛癌中性粒细胞建模](Neutrophil profiling illuminates anti-tumor antigen presenting potency.md) 
    
-   :simple-materialformkdocs:{ .lg .middle } __CV&CPath__

    ---

    - [:octicons-arrow-right-24:ViT原理以及代码实现](Vision Transformer (一).md)
    - [:octicons-arrow-right-24:下一代计算病理学编码框架](Towards a general-purpose foundation model for computational pathology.md)   
    - [:octicons-arrow-right-24:MyGo](Mygo.md)
    - [:octicons-arrow-right-24:使用多种模态数据进行预后预测](Modeling Dense Multimodal Interactions Between Biological Pathways and Histology for Survival Prediction.md)

-   :material-format-font:{ .lg .middle } __结果可视化__

    ---

    - [:octicons-arrow-right-24:可视化范例：以CLAM为例](Visualization Work Flow.md)
    - [:octicons-arrow-right-24:不止于注意力分数](Transformer Interpretability Beyond Attention Visualization.md)
    - [:octicons-arrow-right-24:代码解读](Attention Generation Work Flow.md)
    

-   :simple-aboutdotme:{ .lg .middle } __Re:0 WSI processing&visualization__

    ---

    - [:octicons-arrow-right-24:预处理](Session 1.ipynb)
    - [:octicons-arrow-right-24:A-toy-model-infer](Session 2.ipynb)   
    - [:octicons-arrow-right-24:解释与可视化](Session 3.ipynb)

</div> -->

<div class="grid">
  <a href="#" class="card">Foo</a>
  <a href="#" class="card">Bar</a>
  <a href="#" class="card">Baz</a>
</div>