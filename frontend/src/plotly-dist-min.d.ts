declare module 'plotly.js-dist-min' {
  export function react(root: HTMLElement, data: object[], layout: object, config?: object): void;
  export function newPlot(root: HTMLElement, data: object[], layout: object, config?: object): void;
  export function purge(root: HTMLElement): void;
}
