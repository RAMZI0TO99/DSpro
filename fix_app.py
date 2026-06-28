import re

with open('Frontend/js/app.js', 'r', encoding='utf-8') as f:
    content = f.read()

start_idx = content.find("            } else if (ext === 'pdf') {")
end_idx = content.find("                size: 16,")

replacement = """            } else if (ext === 'pdf') {
                mainPdfEl.style.display = 'block';
            } else {
                mainVideoEl.style.display = 'block';
            }
        }
    });

    tabGraph.addEventListener('click', () => {
        tabGraph.style.background = 'var(--accent)';
        tabGraph.style.color = 'white';
        tabGraph.style.border = 'none';
        
        tabPlayer.style.background = 'var(--bg-base)';
        tabPlayer.style.color = 'var(--text-primary)';
        tabPlayer.style.border = '1px solid var(--border)';
        
        mainVideoEl.style.display = 'none';
        mainImageEl.style.display = 'none';
        mainPdfEl.style.display = 'none';
        graphView.style.display = 'block';
        
        if (!network) {
            loadGraph();
        }
    });
}

if (refreshGraphBtn) {
    refreshGraphBtn.addEventListener('click', loadGraph);
}

async function loadGraph() {
    try {
        const container = document.getElementById('graph-network');
        container.innerHTML = '<div style="display:flex; justify-content:center; align-items:center; height:100%; color:var(--text-secondary);">Loading Graph...</div>';
        
        let queryParams = "";
        if (activeFolderId && activeFolderId !== "All Media") {
            const targetIds = Object.keys(folderLibrary).filter(id => folderLibrary[id] === activeFolderId);
            if (targetIds.length > 0) {
                queryParams = "?target_video_ids=" + targetIds.join(',');
            }
        }

        const res = await fetch("/graph" + queryParams);
        const data = await res.json();
        
        if (!data.nodes || data.nodes.length === 0) {
            container.innerHTML = "<div style=\\"display:flex; justify-content:center; align-items:center; height:100%; color:var(--text-secondary);\\">No Graph Data Available for " + (activeFolderId || 'All Media') + ".</div>";
            return;
        }

        const nodes = new vis.DataSet(data.nodes.map(n => ({
            id: n.id,
            label: n.label,
            group: n.group,
            title: `Type: ${n.group}`
        })));

        const edges = new vis.DataSet(data.edges.map(e => ({
            from: e.from,
            to: e.to,
            label: e.label,
            font: { align: 'middle', size: 10, color: 'rgba(255,255,255,0.7)' }
        })));

        const graphData = { nodes, edges };
        const options = {
            nodes: {
                shape: 'dot',
"""

new_content = content[:start_idx] + replacement + content[end_idx:]
with open('Frontend/js/app.js', 'w', encoding='utf-8') as f:
    f.write(new_content)
print("Done")
