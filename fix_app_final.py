import re

with open('Frontend/js/app.js', 'r', encoding='utf-8') as f:
    content = f.read()

# I will find the exact last known good part of the code before the mess up.
# The mess up happened after line 1123 `const tabPlayer = document.getElementById('tab-player');`
# Let's replace everything after `bootSystem();`

start_idx = content.find('bootSystem();') + len('bootSystem();')

replacement = """

// ============================================================
// KNOWLEDGE GRAPH VISUALIZATION
// ============================================================
const tabPlayer = document.getElementById('tab-player');
const tabGraph = document.getElementById('tab-graph');
const graphView = document.getElementById('graph-view');
const refreshGraphBtn = document.getElementById('refresh-graph-btn');
const mainVideoEl = document.getElementById('main-video');
const mainImageEl = document.getElementById('main-image');
const mainPdfEl = document.getElementById('main-pdf');

let network = null;

if (tabPlayer && tabGraph) {
    tabPlayer.addEventListener('click', () => {
        tabPlayer.style.background = 'var(--accent)';
        tabPlayer.style.color = 'white';
        tabPlayer.style.border = 'none';
        
        tabGraph.style.background = 'var(--bg-base)';
        tabGraph.style.color = 'var(--text-primary)';
        tabGraph.style.border = '1px solid var(--border)';
        
        graphView.style.display = 'none';
        
        // Restore whatever was playing
        if (activeVideoId) {
            const ext = videoLibrary[activeVideoId].split('.').pop().toLowerCase();
            if (['jpg', 'jpeg', 'png', 'webp'].includes(ext)) {
                mainImageEl.style.display = 'block';
            } else if (ext === 'pdf') {
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
            container.innerHTML = `<div style="display:flex; justify-content:center; align-items:center; height:100%; color:var(--text-secondary);">No Graph Data Available for ${activeFolderId || 'All Media'}.</div>`;
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
                size: 16,
                font: { size: 12, color: '#ffffff' },
                borderWidth: 2,
                shadow: true
            },
            edges: {
                width: 1,
                color: { inherit: 'from', opacity: 0.6 },
                smooth: { type: 'continuous' }
            },
            physics: {
                forceAtlas2Based: { gravitationalConstant: -26, centralGravity: 0.005, springLength: 230, springConstant: 0.18 },
                maxVelocity: 146,
                solver: 'forceAtlas2Based',
                timestep: 0.35,
                stabilization: { iterations: 150 }
            },
            groups: {
                Person: { color: { background: '#ef4444', border: '#b91c1c' } },
                Organization: { color: { background: '#3b82f6', border: '#1d4ed8' } },
                Location: { color: { background: '#10b981', border: '#047857' } },
                Concept: { color: { background: '#f59e0b', border: '#b45309' } }
            },
            interaction: { hover: true, tooltipDelay: 200 }
        };

        container.innerHTML = '';
        network = new vis.Network(container, graphData, options);
        
    } catch (e) {
        console.error("Failed to load graph", e);
        document.getElementById('graph-network').innerHTML = '<div style="display:flex; justify-content:center; align-items:center; height:100%; color:var(--danger);">Error loading graph.</div>';
    }
}
"""

new_content = content[:start_idx] + replacement
with open('Frontend/js/app.js', 'w', encoding='utf-8') as f:
    f.write(new_content)
print("Done")
